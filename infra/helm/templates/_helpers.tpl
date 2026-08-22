{{/*
Standard chart name/fullname/labels helpers - the same shape `helm create`
scaffolds, kept here so every template can share one naming convention.
*/}}

{{- define "catalogmind.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "catalogmind.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "catalogmind.labels" -}}
app.kubernetes.io/name: {{ include "catalogmind.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "catalogmind.selectorLabels" -}}
app.kubernetes.io/name: {{ include "catalogmind.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Postgres password, resolved exactly once per release and reused everywhere
it's needed (the postgres Secret itself, and the API's POSTGRES_DSN in
api-secret.yaml). Resolving it independently in two places would generate two
different random passwords that silently don't match, breaking the API's DB
connection - see the plan's discussion in PROGRESS.md's Day 6 entry.

**A first version of this got that exact bug wrong**, worth recording:
`randAlphaNum` is not memoized across separate `{{ include }}` call sites - two
calls to this define, one from postgres-statefulset.yaml and one from
api-secret.yaml, each independently fell through to `randAlphaNum 20` and
produced two different strings, confirmed by actually rendering the chart
(`helm template`) and diffing the two base64 secret values, not assumed correct
from reading the template alone. Fixed below by caching the resolved value into
`.Values.postgres.password` the first time it's computed (Helm's `.Values` is a
mutable map shared across every template in one render) - the second call sees
it already set and returns immediately, without generating a second value.

Precedence: an explicit `.Values.postgres.password` (or the cached value from
this define's own first call) wins; otherwise reuse whatever the existing
installed Secret already holds (`lookup`, so a `helm upgrade` never rotates a
running password out from under itself); otherwise generate a fresh random one
for a genuine first install.

`lookup` always returns an empty dict during `helm template`/`helm lint` -
there is no live cluster to query - so this deliberately falls through to
`randAlphaNum` in that case. That's expected: those commands only need the
chart to render, not a stable password across renders - and the memoization
above still guarantees both secrets agree within that one render.
*/}}
{{- define "catalogmind.postgresPassword" -}}
{{- if not .Values.postgres.password -}}
{{- $secretName := printf "%s-postgres" (include "catalogmind.fullname" .) -}}
{{- $existing := lookup "v1" "Secret" .Release.Namespace $secretName -}}
{{- if $existing -}}
{{- $_ := set .Values.postgres "password" (index $existing.data "postgres-password" | b64dec) -}}
{{- else -}}
{{- $_ := set .Values.postgres "password" (randAlphaNum 20) -}}
{{- end -}}
{{- end -}}
{{- .Values.postgres.password -}}
{{- end -}}
