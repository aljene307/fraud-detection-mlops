# Image du scoreur de fraude.
#
# Construction :  docker compose build
# Ou directement : docker build -f docker/Dockerfile -t fraud-scorer .
#                  (le point final : le CONTEXTE est la racine du projet, pas
#                   le dossier docker/, car on a besoin de src/ et du lock)
#
# La meme image sert aussi au service mlflow du docker-compose, avec une
# commande differente : cela garantit que le client et le serveur MLflow ont
# exactement la meme version -- une source classique de bugs subtils.

# slim-trixie et non slim-bookworm : trixie est la variante slim courante.
# Et surtout PAS alpine : alpine utilise musl au lieu de glibc, or les wheels
# de numpy, scipy et xgboost sont compiles pour glibc (manylinux). Sur alpine,
# pip recompilerait depuis les sources -- builds de 20 minutes et bugs
# numeriques subtils.
FROM python:3.13-slim-trixie

# PYTHONUNBUFFERED : sans lui les logs restent coinces dans un tampon et un
#   conteneur qui plante n'affiche RIEN dans `docker logs`.
# PYTHONDONTWRITEBYTECODE : les .pyc sont inutiles dans une image immuable.
# PYTHONPATH : rend `src` importable sans installation. On evite ainsi que
#   src/ atterrisse dans site-packages, ce qui deplacerait PROJECT_ROOT (calcule
#   depuis __file__ dans src/config.py) vers un dossier sans rapport.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---------------------------------------------------------------------------
# Couche 1 : les dependances. Change rarement, donc EN HAUT.
# ---------------------------------------------------------------------------
# C'est tout l'interet de copier le lock SEUL avant le code : tant qu'aucune
# dependance ne bouge, Docker reutilise cette couche en cache. Copier le code
# d'abord ferait reinstaller 130 paquets a chaque correction de typo.
#
# Depuis requirements.lock.txt et non `pip install .` : le lock fige l'arbre
# transitif COMPLET, donc deux constructions a six mois d'ecart donnent la meme
# image.
#
# Ce lock ne contient QUE la cloture runtime : ni pytest, ni ruff, ni httpx2.
# Ce n'est pas automatique -- un `pip freeze` depuis le venv local les
# embarquerait, puisqu'il est installe avec ".[dev]". C'est
# scripts/freeze_lock.py qui calcule la cloture et les ecarte (12 paquets), et
# qui pose au passage le marqueur sys_platform sur pywin32, absent de Linux.
#
# --no-cache-dir : pip conserve un cache de telechargement, inutile dans une
# image et lourd de plusieurs centaines de Mo.
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

# ---------------------------------------------------------------------------
# Couche 2 : le code. Change souvent, donc EN BAS.
# ---------------------------------------------------------------------------
COPY src/ ./src/

# ---------------------------------------------------------------------------
# Bornage des reessais MLflow
# ---------------------------------------------------------------------------
# Place APRES le pip install a dessein : cette instruction ne doit pas
# invalider la couche de dependances, qui coute 11 minutes a reconstruire.
#
# Sans ce bornage, le demarrage degrade ne fonctionne PAS quand le serveur de
# tracking est injoignable. Mesure faite : le conteneur reste bloque sur
# "Waiting for application startup", port FERME, pendant plusieurs minutes --
# MLflow reessaie 7 fois avec un backoff de facteur 2 et un timeout de 120 s.
#
# Or c'est precisement le scenario contre lequel le demarrage degrade a ete
# concu : /health doit repondre 200 pour qu'une sonde de liveness ne tue pas un
# conteneur parfaitement vivant. Un port ferme pendant cinq minutes produit
# exactement le CrashLoopBackOff qu'on voulait eviter.
#
# Avec ces valeurs, l'echec est constate en ~35 s au pire, bien sous la
# start-period de 90 s du HEALTHCHECK.
ENV MLFLOW_HTTP_REQUEST_MAX_RETRIES=2 \
    MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR=1 \
    MLFLOW_HTTP_REQUEST_TIMEOUT=10

# ---------------------------------------------------------------------------
# Utilisateur non privilegie
# ---------------------------------------------------------------------------
# Par defaut un conteneur tourne en root : une faille dans l'API donnerait les
# droits root DANS le conteneur, et les evasions de conteneur existent. UID
# explicite pour que le securityContext de la Phase 5 puisse y correspondre --
# beaucoup de clusters imposent runAsNonRoot et refuseraient une image root.
#
# /mlflow est cree ici, possede par appuser : quand docker-compose y montera un
# volume nomme vide, Docker en heritera les permissions. Sans ca, le service
# mlflow (qui partage cette image) ne pourrait pas ecrire sa base.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /mlflow \
    && chown -R appuser:appuser /mlflow /app

USER appuser

# Documentation uniquement : EXPOSE ne publie RIEN. C'est `docker run -p` ou la
# section ports: du compose qui publie.
EXPOSE 8000

# Docker execute cette commande dans le conteneur et marque celui-ci unhealthy
# apres plusieurs echecs.
#
# start-period=90s est le parametre qui compte, et c'est notre mesure qui le
# dicte : le chargement du modele prend 5 a 14 s (plus le telechargement depuis
# MinIO), PORT FERME. Sans periode de grace, la sonde echouerait des la premiere
# seconde et le conteneur naitrait unhealthy. C'est l'equivalent Docker du
# startupProbe Kubernetes.
#
# Sur /health et non /ready : liveness, pas readiness. Un conteneur dont le
# modele n'a pas charge est vivant, il attend que MLflow revienne.
#
# En python et non curl : l'image slim n'embarque pas curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# --host 0.0.0.0 et NON 127.0.0.1 : dans un conteneur, ecouter sur localhost
# signifie "seul le conteneur peut me joindre" et le mappage de port ne verrait
# rien. C'est l'erreur numero un du premier conteneur.
#
# Forme JSON (exec) et non forme shell : le processus devient PID 1 et recoit
# directement le SIGTERM de `docker stop`, donc le lifespan s'arrete proprement.
# En forme shell, un /bin/sh s'interpose, avale le signal, et Docker tue tout au
# bout de 10 s.
#
# Un seul worker : prometheus_client tient ses compteurs par processus, donc
# plusieurs workers donneraient des scrapes partiels. On met a l'echelle par
# replicas, pas par workers.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
