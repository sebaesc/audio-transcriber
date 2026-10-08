# Resultados de la evaluación sobre videos de clase con ground truth propio

Informe técnico sobre el reporte `eval/custom/large-v3.json` (generado el 2026-10-08).
Las definiciones de cada métrica están en `docs/evaluation.md` y en `python -m audio_transcriber.evaluation --metrics`.

## 1. Resumen

Se evaluó `faster-whisper large-v3` sobre 7 recortes de ~20 min de videos de clase, contra
transcripciones escritas a mano por el equipo (ground truth propio, con hablante e idioma por segmento).

- **WER medio 0.337** (0.324 agregando las palabras de los 7 recortes; mediana 0.304). Rango muy amplio: de 0.173 a 0.491.
- **El idioma es el principal factor de error.** Los dos recortes con más español (Level 2 Week 7, 50% del
  texto de referencia; 1930 Week 1, 26%) tienen los peores WER (0.491 y 0.468) y los peores errores de mezcla de idiomas.
- **La etiqueta de idioma por segmento es útil pero sobre-etiqueta español**: precisión media 0.873 y, en
  6 de los 7 recortes la parte de texto en español que mide el reporte es entre 1.2 y 4 veces la de la referencia.
- **La diarización asigna bien al hablante cuando detecta voz** (confusión media 0.092), pero el DER medio
  (0.549) está dominado por `missed`, que en buena parte es un artefacto de cómo se miden los límites de segmento.
- El pipeline **omite** palabras (retenido medio 0.882), en general más de lo que **inventa**, salvo en un recorte.

## 2. Configuración y datos

| Parámetro | Valor |
| --------- | ----- |
| Backend / modelo | `faster-whisper` 1.2.1 / `large-v3`, GPU (`cuda`) |
| Idioma | Detección por tramos de 15 s entre `en` y `es` (umbral 0.5) |
| Decodificación | `beam_size=5`, `best_of=5`, VAD activado, `temperature` en cascada 0.0–1.0, `seed=0` |
| Diarización | `pyannote/speaker-diarization-community-1` (pyannote.audio 4.0.7), asignación de hablante por palabra |
| Recortes | 7 videos, 19.7–22.6 min c/u (~2.4 h en total) |
| Ground truth | Segmentos con hablante, idioma y texto; tiempos del video completo, con `clip_start` por archivo cuando el recorte no empieza en 0 |

Seis de los siete recortes usan `clip_start` (4 s a 1199 s). El de `Level 1 1101055 Week 15` empieza en 0.
El reporte no tiene tiempos de ejecución (`elapsed_s = 0`): son transcripciones puntuadas a posteriori,
así que **este informe no dice nada sobre velocidad**. En una corrida anterior de un recorte
(`week8-clip`, ver §6) el factor de tiempo real fue 4.2×.

## 3. Resultados por recorte

### 3.1 Texto

| Recorte | Palabras ref. | % español ref. | WER | Retenido | Cobertura | Err. mezcla idioma |
| ------- | ------------: | -------------: | --: | -------: | --------: | -----------------: |
| 1930 · 1101155 · Week 1 | 894 | 25.7% | 0.468 | 0.864 | 0.922 | 0.258 |
| 1930 · 1108028 · Week 3 | 1401 | 2.7% | 0.246 | 0.843 | 0.892 | 0.020 |
| Level 1 · 1101055 · Week 15 | 1604 | 11.3% | 0.256 | 0.879 | 0.915 | 0.071 |
| Level 1 · 1101251 · Week 8 | 1498 | 13.7% | 0.419 | 0.891 | 1.086 | 0.036 |
| Level 2 · 1101058 · Week 7 | 1301 | 50.3% | 0.491 | 0.875 | 1.015 | 0.159 |
| Level 2 · 1103119 · Week 14 | 1567 | 17.0% | 0.173 | 0.923 | 0.960 | 0.036 |
| Level 3 · 1103119 · Week 3 | 1409 | 4.5% | 0.304 | 0.902 | 0.956 | 0.134 |
| **Media** | | | **0.337** | **0.882** | **0.964** | **0.102** |

### 3.2 Composición del error (sobre palabras de referencia)

| Recorte | Sustituciones | Borrados | Inserciones |
| ------- | ------------: | -------: | ----------: |
| 1930 · Week 1 | 27.3% | 13.6% | 5.8% |
| 1930 · Week 3 | 4.1% | 15.7% | 4.9% |
| Level 1 · Week 15 | 9.9% | 12.1% | 3.6% |
| Level 1 · Week 8 | 11.5% | 10.9% | **19.5%** |
| Level 2 · Week 7 | 22.5% | 12.5% | 14.1% |
| Level 2 · Week 14 | 5.9% | 7.7% | 3.7% |
| Level 3 · Week 3 | 15.2% | 9.8% | 5.4% |
| **Total (9674 palabras)** | **12.7%** | **11.6%** | **8.1%** |

### 3.3 Idioma por segmento (tiempo de habla puntuado)

| Recorte | Exactitud | Recall `en` | Recall `es` | `en` etiquetado `es` (s) | `es` etiquetado `en` (s) |
| ------- | --------: | ----------: | ----------: | -----------------------: | -----------------------: |
| 1930 · Week 1 | 0.721 | 0.690 | 0.957 | 145.1 | 2.7 |
| 1930 · Week 3 | 0.985 | 0.985 | 0.943 | 11.0 | 0.4 |
| Level 1 · Week 15 | 0.853 | 0.852 | 0.886 | 124.8 | 2.0 |
| Level 1 · Week 8 | 0.988 | 1.000 | 0.000 | 0.0 | 12.0 |
| Level 2 · Week 7 | 0.805 | 0.751 | 0.960 | 90.8 | 5.1 |
| Level 2 · Week 14 | 0.853 | 0.853 | 0.860 | 144.6 | 5.2 |
| Level 3 · Week 3 | 0.908 | 0.908 | n/a | 79.0 | 0.0 |
| **Media** | **0.873** | | | | |

### 3.4 Diarización

| Recorte | DER | Missed | False alarm | Confusión de hablante |
| ------- | --: | -----: | ----------: | --------------------: |
| 1930 · Week 1 | 0.704 | 0.563 | 0.005 | 0.136 |
| 1930 · Week 3 | 0.427 | 0.382 | 0.000 | 0.045 |
| Level 1 · Week 15 | 0.367 | 0.266 | 0.014 | 0.087 |
| Level 1 · Week 8 | 0.468 | 0.329 | 0.010 | 0.130 |
| Level 2 · Week 7 | 0.764 | 0.440 | 0.143 | 0.181 |
| Level 2 · Week 14 | 0.534 | 0.500 | 0.013 | 0.022 |
| Level 3 · Week 3 | 0.581 | 0.476 | 0.063 | 0.042 |
| **Media** | **0.549** | | | **0.092** |

## 4. Análisis

### 4.1 El español es lo que más duele en el texto

El WER sigue de cerca la proporción de español en la referencia. Los dos recortes con más español
tienen los peores WER, y en ambos pesan las **sustituciones** (27% y 22% de las palabras de referencia,
contra 4–15% en el resto): es el patrón que ya documentamos con Whisper, donde el español dicho en un
contexto mayormente inglés sale transliterado a ortografía inglesa o decodificado en el idioma equivocado
(ver [[mixed-language-audio]] en las notas del proyecto). Hay que decir que la correlación es sobre 7
puntos y no es perfecta: `Level 3 · Week 3` tiene solo 4.5% de español y aun así un error de mezcla de
0.134 (el modelo cree que hay 17.8% de español), y `Level 1 · Week 8` tiene WER 0.419 con un 13.7% de español.

### 4.2 Sobre-etiquetado de español

En 6 de 7 recortes el modelo etiqueta como `es` más tiempo del que corresponde:
la columna "`en` etiquetado `es`" va de 79 a 145 s en cinco de los siete recortes, contra solo 0–12 s en la dirección inversa.
El **recall de `es` es alto (0.86–0.96)** y el de `en` es el que cae (0.69–0.91 en cinco recortes).
Es decir, la detección por tramos de 15 s rara vez deja pasar español, pero marca español en tramos
de inglés. Probable causa, no verificada: hablantes con acento hispano en inglés y tramos cortos o con
ruido, donde el detector de idioma de Whisper tiene poca evidencia. Las filas con `es` = `n/a` o 0.000 de
recall (`Level 3 Week 3`, `Level 1 Week 8`) no son comparables: casi no hay español etiquetado en la
referencia (0 s y 12 s), así que el recall de `es` no se puede calcular de forma fiable.

### 4.3 Diarización: confusión baja, DER alto

La **confusión de hablante** es baja (media 0.092; 0.022–0.181): cuando el sistema transcribe habla,
generalmente se la atribuye al hablante correcto. El **DER** (0.549) lo domina `missed` (0.27–0.56), y
por el diseño de la métrica no es una medida pura de diarización: puntúa los segmentos del transcript
(no los turnos crudos de pyannote) y los tiempos del ground truth abarcan pausas que Whisper no incluye
(ver `docs/evaluation.md`, sección "Writing a ground truth": tiempos escritos al segundo son más gruesos que el collar
de 0.25 s). Además, `missed` incluye habla que no se transcribió: el recorte con mayor `missed` (1930 Week 1,
0.563) es también el de menor palabras por minuto (45) y con 14% de borrados, lo que sugiere
habla real no transcrita, no solo un problema de límites.

Dos casos especiales:

- **Level 2 · Week 7** tiene el peor DER (0.764), el único `false_alarm` relevante (0.143) y la mayor confusión
  (0.181). Es el recorte con más español y más inserciones (14.1%): hay segmentos que el modelo produce donde la
  referencia no tiene habla.
- **Level 2 · Week 14** reporta `reference_seconds = 1444.5` en un recorte de 1200 s. Es posible porque el
  ground truth permite habla solapada, y se cuenta el tiempo de cada hablante; no es un error.
  Tiene la confusión más baja (0.022).

### 4.4 Omisiones e inserciones

Globalmente se omiten más palabras (11.6%) de las que se inventan (8.1%), y el "retenido" medio es 0.882:
casi 12 de cada 100 palabras dichas no aparecen como nada. La excepción es **Level 1 · Week 8**, con
inserciones de 19.5% y cobertura de 1.086. En ese caso el modelo produce más texto del que hay en la
referencia (317 segmentos, el segundo mayor número). Puede ser alucinación o repetición, o también habla
real que el ground truth no transcribió; no se inspeccionó el texto, y habría que hacerlo con el diff de palabras
descrito en `docs/evaluation.md` antes de concluir. Además solo detectó `en` en ese recorte, aunque la referencia
tiene 13.7% de español.

## 5. Conclusiones

1. Con la configuración actual, el texto es **utilizable pero no fiable** en clases con mucho español: con >25% de español
   el WER ronda 0.47–0.49. En clases mayormente inglesas el WER baja a 0.17–0.30.
2. La **etiqueta `spoken_language` por segmento** hoy es mejor para "detectar que hay español" que para
   decir "este segmento es inglés". Un consumidor que filtre por `es` obtendrá falsos positivos frecuentes.
3. La **etiqueta de hablante** es razonablemente confiable donde hay habla transcrita; la métrica de DER
   total no sirve para comparar diarizadores con este ground truth mientras los tiempos sean de resolución al segundo.
4. La variación entre recortes (WER 0.17–0.49) es mayor que cualquier diferencia que se espere entre
   configuraciones cercanas: **con 7 recortes de una sola corrida, no se pueden sacar conclusiones finas**.
   Sirve para ver dónde falla, y no para ordenar modelos de rendimiento parecido.

## 6. Limitaciones y trabajo siguiente

- **Una sola configuración y una sola corrida** (`seed=0`). No hay comparación con otro modelo ni con ElevenLabs Scribe todavía;
  si se hace, puntuar transcripciones ya generadas (`--transcripts`) y cuidar el presupuesto de créditos
  (~66.7 créditos por minuto de audio; los 7 recortes suman ≈ 144 min ≈ 9 600 créditos, casi toda una clave gratuita).
- **Los números absolutos son pesimistas**: el ground truth incluye muletillas que Whisper omite y los tiempos del
  ground truth están al segundo.
- **No hay anotación de calidad del ground truth** (un solo anotador, sin medida de acuerdo). Parte del WER puede ser
  discrepancia de transcripción, especialmente en español o habla poco clara.
- **Hay un reporte anterior** de otro recorte de Week 8 (`19.53-40.09`, `eval/custom/week8-clip-large-v3.local.json`,
  no versionado): WER 0.266, exactitud de idioma 0.998, DER 0.482. Es otro tramo del mismo video que el de
  `11.04-31.04` (WER 0.419): el mismo video produce 0.27 y 0.42 según el tramo, otro indicio de la varianza entre recortes.
- Próximos pasos razonables: inspeccionar con el diff de palabras `Level 1 Week 8` (inserciones) y `1930 Week 1`
  (sustituciones y borrados), probar un umbral de idioma más alto o tramos más largos para reducir el sobre-etiquetado
  de español, y escribir los tiempos del ground truth con décimas en un recorte para medir el DER sin el artefacto de resolución.
