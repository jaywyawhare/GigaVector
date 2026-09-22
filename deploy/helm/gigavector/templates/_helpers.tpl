{{- define "gigavector.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "gigavector.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "gigavector.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "gigavector.labels" -}}
app.kubernetes.io/name: {{ include "gigavector.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "gigavector.selectorLabels" -}}
app.kubernetes.io/name: {{ include "gigavector.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "gigavector.image" -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) -}}
{{- end -}}
