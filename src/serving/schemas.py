"""Contrat de l'API : ce que le service accepte et ce qu'il renvoie.

Separe de app.py pour qu'on puisse lire le contrat d'un seul coup d'oeil, sans
traverser la logique des endpoints.

**Pydantic** valide des donnees a partir d'annotations de types Python. On
declare une classe a champs types ; FastAPI verifie chaque JSON entrant contre
elle AVANT d'appeler notre fonction, et repond 422 en nommant le champ fautif.

Pourquoi ces champs sont ecrits EN DUR ici, alors que le step 2.A lit les
colonnes depuis la signature du modele ? Parce que ce sont deux roles opposes :

    on s'ADAPTE a ce qu'on CONSOMME  -> les colonnes viennent du modele deploye
    on reste STABLE pour ce qu'on EXPOSE -> les champs d'API viennent du code

Un contrat public ne doit pas muter parce qu'on a promu un modele : ca casserait
tous les clients sans le moindre avertissement. Le lien entre les deux est un
controle de coherence au demarrage (voir app.py) : en cas de desaccord, le
service demarre degrade et le DIT, au lieu de laisser la contradiction se
resoudre en silence a la premiere requete.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Plafond du lot : borne la taille du corps de requete et le travail par appel.
MAX_BATCH_SIZE = 1_000

# allow_inf_nan=False sur TOUS les flottants : un NaN traverserait sinon jusqu'au
# modele, qui produirait un score sans aucune valeur, sans lever d'erreur.
Finite = Annotated[float, Field(allow_inf_nan=False)]

LogAmount = Annotated[
    float,
    Field(
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "log1p(Amount). Toujours >= 0 puisque Amount l'est. Cette borne "
            "attrape un client qui enverrait le montant brut a la place."
        ),
    ),
]

HourOfDay = Annotated[
    float,
    Field(
        ge=0.0,
        le=23.0,
        allow_inf_nan=False,
        description="Heure de la journee, (Time // 3600) %% 24. Attrape un modulo oublie.",
    ),
]

# Une vraie fraude du split de test : rend /docs immediatement utilisable.
EXAMPLE_TRANSACTION: dict[str, float] = {
    "V1": 0.315642, "V2": 1.636778, "V3": -1.51965, "V4": 4.028571,
    "V5": -1.186794, "V6": -0.789813, "V7": -2.279807, "V8": 0.472988,
    "V9": -1.657635, "V10": -2.89499, "V11": 1.601985, "V12": -2.824946,
    "V13": 1.269204, "V14": -5.591364, "V15": -0.974827, "V16": -2.737795,
    "V17": -4.961534, "V18": -2.224797, "V19": -0.136117, "V20": 0.388885,
    "V21": 0.345921, "V22": -0.108002, "V23": -0.165442, "V24": 0.279895,
    "V25": 0.808783, "V26": 0.117363, "V27": 0.589595, "V28": 0.309064,
    "log_amount": 1.56653, "hour_of_day": 19.0,
}


class TransactionIn(BaseModel):
    """Une transaction a scorer : 28 composantes PCA + 2 variables derivees.

    ``extra="forbid"`` rejette tout champ inconnu. C'est volontairement strict :
    accepter puis ignorer silencieusement un champ laisserait croire au client
    qu'il envoie quelque chose qui compte. Un client qui envoie ``Amount`` au
    lieu de ``log_amount`` obtient ainsi les DEUX informations d'un coup : le
    champ manquant et le champ en trop.

    Aucune borne sur V1-V28 : ce sont des composantes PCA non bornees, et une
    borne arbitraire rejetterait les transactions extremes -- or c'est
    precisement la que vit la fraude.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": EXAMPLE_TRANSACTION},
    )

    V1: Finite
    V2: Finite
    V3: Finite
    V4: Finite
    V5: Finite
    V6: Finite
    V7: Finite
    V8: Finite
    V9: Finite
    V10: Finite
    V11: Finite
    V12: Finite
    V13: Finite
    V14: Finite
    V15: Finite
    V16: Finite
    V17: Finite
    V18: Finite
    V19: Finite
    V20: Finite
    V21: Finite
    V22: Finite
    V23: Finite
    V24: Finite
    V25: Finite
    V26: Finite
    V27: Finite
    V28: Finite
    log_amount: LogAmount
    hour_of_day: HourOfDay


# Le contrat expose, sous forme comparable a bundle.feature_columns.
TRANSACTION_FIELDS: tuple[str, ...] = tuple(TransactionIn.model_fields)


class PredictionOut(BaseModel):
    """Reponse a une transaction.

    Elle porte la version du modele ET le seuil utilise : une prediction passee
    doit rester auditable. ``threshold_source`` rend visible qu'un seuil vient
    d'une variable d'environnement plutot que du modele -- une surcharge
    oubliee ne doit jamais s'appliquer en silence.

    ``protected_namespaces=()`` : Pydantic v2 reserve le prefixe ``model_`` pour
    ses propres attributs et avertit sinon. On le libere ici, ``model_name`` et
    ``model_version`` etant les noms justes pour un consommateur d'API.
    """

    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float = Field(
        ...,
        description=(
            "Score dans [0, 1]. ATTENTION : ce n'est pas une probabilite "
            "calibree -- scale_pos_weight repondere la fonction de cout, donc "
            "le modele repond 'probabilite si la fraude representait 50 % du "
            "trafic'. A n'utiliser que compare au seuil."
        ),
    )
    is_fraud: bool = Field(..., description="fraud_probability >= threshold")
    threshold: float
    threshold_source: str = Field(..., description="'mlflow_run' ou 'env'")
    model_name: str
    model_version: str


class BatchPredictionOut(BaseModel):
    """Une ligne de resultat dans un lot."""

    index: int = Field(..., description="Position dans la liste envoyee")
    fraud_probability: float
    is_fraud: bool


class BatchIn(BaseModel):
    """Un lot de transactions.

    Le lot existe pour une raison mesuree : scorer 1 ligne et 200 lignes coute
    le meme temps (~8 ms), le cout etant fixe par appel. Le lot divise donc le
    cout par ligne par ~200.
    """

    model_config = ConfigDict(extra="forbid")

    transactions: list[TransactionIn] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"Entre 1 et {MAX_BATCH_SIZE} transactions.",
    )


class BatchOut(BaseModel):
    """Reponse a un lot.

    Les metadonnees du modele figurent UNE fois, pas repetees a chaque ligne :
    elles sont identiques pour tout le lot, puisqu'un seul modele l'a score.
    """

    model_config = ConfigDict(protected_namespaces=())

    count: int
    threshold: float
    threshold_source: str
    model_name: str
    model_version: str
    predictions: list[BatchPredictionOut]


class HealthOut(BaseModel):
    """Reponse de /health. Volontairement minimale : cet endpoint ne doit rien
    savoir du modele."""

    status: str = "ok"
