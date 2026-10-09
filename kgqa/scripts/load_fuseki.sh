#!/usr/bin/env bash
# Create the "alarms" dataset on Fuseki (idempotent) and load the ontology.
set -euo pipefail

FUSEKI="${FUSEKI_URL:-http://localhost:3030}"
AUTH="admin:${FUSEKI_ADMIN_PASSWORD:-admin}"
DATASET="${DATASET:-alarms}"
TTL="${1:-ontology/alarm_ontology.ttl}"

echo "waiting for Fuseki at $FUSEKI ..."
for _ in $(seq 1 30); do
  curl -fsS "$FUSEKI/\$/ping" >/dev/null 2>&1 && break
  sleep 1
done

code=$(curl -s -o /dev/null -w "%{http_code}" -u "$AUTH" -X POST \
  "$FUSEKI/\$/datasets" --data "dbName=$DATASET&dbType=tdb2")
case "$code" in
  200|201) echo "dataset $DATASET created" ;;
  409)     echo "dataset $DATASET already exists" ;;
  *)       echo "dataset creation failed (HTTP $code)"; exit 1 ;;
esac

# PUT replaces the default graph, so reloading never duplicates triples.
curl -fsS -u "$AUTH" -X PUT -H "Content-Type: text/turtle" \
  --data-binary "@$TTL" "$FUSEKI/$DATASET/data?default" >/dev/null
echo "loaded $TTL"

n=$(curl -fsS -H "Accept: application/sparql-results+json" \
  --data-urlencode "query=SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }" \
  "$FUSEKI/$DATASET/query" | python3 -c "import json,sys; print(json.load(sys.stdin)['results']['bindings'][0]['n']['value'])")
echo "triples in $DATASET: $n"
echo "export SPARQL_ENDPOINT=$FUSEKI/$DATASET/query"
