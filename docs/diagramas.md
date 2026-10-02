# Diagramas del repositorio

Cinco vistas de Audio Transcriber: el flujo de una ejecución, el modelo de clases (UML), los casos
de uso, la secuencia de llamadas y la evaluación contra el ground truth. Los diagramas están en Mermaid, que GitHub y la mayoría de editores renderizan directamente.

## 1. Diagrama de flujo — una ejecución de `audio-transcriber`

Recorre `cli.main` → `pipeline.run` → `pipeline._transcribe_one`. Un fallo por archivo nunca
aborta el lote; un fallo de configuración aborta antes de escribir nada.

```mermaid
flowchart TD
    A([Inicio: audio-transcriber -i ENTRADA -b BACKEND -o SALIDA]) --> B{¿--list-backends<br/>o --version?}
    B -- Sí --> B1[Imprimir y salir] --> Z0([Exit 0])
    B -- No --> C{¿Faltan --input<br/>o --backend?}
    C -- Sí --> Z3([Exit 3: CONFIG_ERROR])
    C -- No --> D[Parsear --option KEY=VALUE<br/>valores leídos como JSON]
    D --> E["registry.create(nombre, **config)"]
    E --> E1{¿Backend registrado<br/>e instalado?}
    E1 -- No --> Z3
    E1 -- Sí --> F["pipeline.validate()"]

    subgraph VAL [Validación previa]
        F --> F1["discovery.discover(entrada, glob)"]
        F1 --> F2{¿Hay archivos de audio?}
        F2 -- Sí --> F3["audio.check_ffmpeg()"]
        F3 --> F4["backend.check_requirements()<br/>modelo, credenciales, HF_TOKEN"]
    end

    F2 -- No --> Z3
    F3 -- ffmpeg ausente --> Z3
    F4 -- Requisito no cumplido --> Z3
    F4 -- OK --> G["prepare_run_directory(out/run-id)<br/>run-id = timestamp UTC"]
    G --> G1{¿Directorio ya<br/>tiene resultados?}
    G1 -- Sí --> Z3
    G1 -- No --> H[Crear Manifest:<br/>backend, model_version, config, tool_version]

    H --> I{¿Quedan archivos?}
    I -- Sí --> J["InputRecord.from_path()<br/>sha256 + tamaño"]
    J --> K["audio.decoded(): ffmpeg → WAV mono 16 kHz"]
    K --> L["backend.transcribe(wav, origen)"]
    L --> M["assert_transcript_conforms()"]
    M -- OK --> N[Escribir &lt;stem&gt;.json<br/>array de segmentos]
    N --> N1[Registrar entrada con metadatos] --> I
    K -- AudioDecodeError --> O[Registrar FileFailure]
    L -- TranscriptionError --> O
    M -- ConformanceError --> O
    O --> I

    I -- No --> P[Escribir manifest.json y errors.json]
    P --> Q{RunResult.status}
    Q -- SUCCEEDED --> Z0b([Exit 0])
    Q -- PARTIAL --> Z1([Exit 1])
    Q -- ABORTED --> Z2([Exit 2])

    L -. Ctrl+C .-> Z130([Exit 130: resultados parciales conservados])
```

### Dentro de `FasterWhisperBackend.transcribe`

```mermaid
flowchart TD
    A[WAV 16 kHz] --> B{language_detection}
    B -- file --> C[model.transcribe archivo completo<br/>idioma único]
    B -- chunked --> D[VAD → speech_chunks]
    D --> E[Detectar idioma por chunk<br/>restringido a languages, p. ej. en/es]
    E --> F[Transcribir cada tramo<br/>con su idioma]
    C --> G[Normalizar a Segment + palabras]
    F --> G
    G --> H{diarization=true?}
    H -- No --> R[Transcript]
    H -- Sí --> I[pyannote: turnos de hablante]
    I --> J{speaker_assignment}
    J -- word --> K[Hablante por palabra,<br/>suavizar y partir segmentos]
    J -- segment --> L[Hablante por segmento completo]
    K --> R
    L --> R
```

## 2. Diagrama UML de clases

```mermaid
classDiagram
    direction LR

    class TranscriptionBackend {
        <<abstract>>
        +name: str
        +capabilities: Capabilities
        +config: dict
        +check_requirements()* None
        +model_version()* str
        +transcribe(audio_path, source)* Transcript
        +effective_config() dict
    }

    class Capabilities {
        <<dataclass frozen>>
        +diarization: bool
        +language_detection: bool
        +word_timestamps: bool
        +translation: bool
        +to_dict() dict
    }

    class FasterWhisperBackend {
        +name = "faster-whisper"
        -_load_model() WhisperModel
        -_transcribe_chunked()
        -_maybe_diarize()
        -_normalize_with_words()
    }

    class ElevenLabsScribeBackend {
        +name = "elevenlabs-scribe"
        -_session() ElevenLabs
        -_request() dict
        -_normalize(response) Transcript
        -_segments()
    }

    class Transcript {
        <<dataclass frozen>>
        +source: str
        +segments: tuple~Segment~
        +language: str?
        +duration: float?
        +backend: str
        +model_version: str
        +text: str
        +to_json_data() list
        +metadata() dict
    }

    class Segment {
        <<dataclass frozen>>
        +start_time: float
        +end_time: float
        +text: str
        +speaker: str?
        +spoken_language: str?
        +duration: float
        +to_dict() dict
    }

    class Manifest {
        <<dataclass>>
        +run_id: str
        +backend: str
        +model_version: str
        +config: dict
        +inputs: list~InputRecord~
        +started_at: str
        +finished_at: str?
        +succeeded: int
        +failed: int
        +to_dict() dict
    }

    class InputRecord {
        <<dataclass frozen>>
        +path: str
        +checksum: str
        +size_bytes: int
        +transcript: str?
        +duration: float?
        +language: str?
        +segment_count: int?
        +from_path(path)$ InputRecord
        +with_result(path, metadata) InputRecord
    }

    class RunResult {
        <<dataclass>>
        +run_directory: Path
        +manifest: Manifest
        +transcripts: list~Transcript~
        +failures: list~FileFailure~
        +status: RunStatus
    }

    class FileFailure {
        <<dataclass frozen>>
        +source: str
        +error: str
        +message: str
    }

    class RunStatus {
        <<enumeration>>
        SUCCEEDED
        PARTIAL
        ABORTED
    }

    class ExitCode {
        <<enumeration>>
        SUCCESS = 0
        PARTIAL = 1
        ABORTED = 2
        CONFIG_ERROR = 3
        INTERRUPTED = 130
    }

    class registry {
        <<module>>
        +register(cls)
        +available() tuple
        +unavailable() dict
        +get(name) type
        +create(name, **config) TranscriptionBackend
    }

    class pipeline {
        <<module>>
        +validate(input, backend, pattern) list~Path~
        +run(input, backend_name, output_root, ...) RunResult
        -_transcribe_one(backend, source) Transcript
    }

    class cli {
        <<module>>
        +main(...)
    }

    class AudioTranscriberError
    class ConfigError
    class UnknownBackendError
    class BackendNotInstalledError
    class BackendRequirementError
    class TranscriptionError
    class AudioDecodeError

    TranscriptionBackend <|-- FasterWhisperBackend
    TranscriptionBackend <|-- ElevenLabsScribeBackend
    TranscriptionBackend *-- Capabilities
    TranscriptionBackend ..> Transcript : produce
    Transcript "1" *-- "0..*" Segment
    Manifest "1" *-- "0..*" InputRecord
    RunResult *-- Manifest
    RunResult o-- "0..*" Transcript
    RunResult *-- "0..*" FileFailure
    RunResult ..> RunStatus
    registry o-- "0..*" TranscriptionBackend : registra clases
    pipeline ..> registry : create()
    pipeline ..> TranscriptionBackend : usa
    pipeline ..> RunResult : devuelve
    cli ..> pipeline : run()
    cli ..> ExitCode

    AudioTranscriberError <|-- ConfigError
    ConfigError <|-- UnknownBackendError
    UnknownBackendError <|-- BackendNotInstalledError
    AudioTranscriberError <|-- BackendRequirementError
    AudioTranscriberError <|-- TranscriptionError
    AudioTranscriberError <|-- AudioDecodeError
```

### Paquetes y dependencias

```mermaid
flowchart LR
    subgraph core [audio_transcriber]
        cli --> pipeline
        pipeline --> registry
        pipeline --> discovery
        pipeline --> audio
        pipeline --> manifest
        pipeline --> conformance
        registry --> backend
        backend --> schema
        conformance --> schema
    end
    subgraph backends [audio_transcriber.backends]
        faster_whisper --> chunking
        faster_whisper --> diarization
        elevenlabs_scribe --> credentials
        diarization --> credentials
    end
    subgraph evaluation [audio_transcriber.evaluation]
        eval_main[__main__] --> runner
        runner --> corpus
        runner --> scoring
        runner --> talkbank
        runner --> annotation
        scoring --> text
    end
    registry -. import perezoso por extra .-> faster_whisper
    registry -. import perezoso por extra .-> elevenlabs_scribe
    runner --> registry
    runner --> audio
```

## 3. Diagrama de casos de uso

Mermaid no tiene un tipo de diagrama de casos de uso; se representa con actores (rectángulos) y
casos de uso (óvalos) dentro del límite del sistema.

```mermaid
flowchart LR
    U["👤 Usuario<br/>(transcribe audio)"]
    D["👤 Desarrollador<br/>(agrega backends)"]
    E["👤 Evaluador<br/>(mide calidad)"]

    FF[["⚙️ ffmpeg"]]
    EL[["☁️ ElevenLabs API"]]
    HF[["☁️ Hugging Face<br/>(pyannote)"]]
    TB[["☁️ Corpus de referencia<br/>+ TalkBank"]]

    subgraph S [Audio Transcriber]
        UC1([Transcribir un directorio o archivo])
        UC2([Elegir backend])
        UC3([Pasar opciones del backend])
        UC4([Detectar idioma por chunk])
        UC5([Diarizar hablantes])
        UC6([Listar backends instalados])
        UC7([Consultar manifest y errores])
        UC8([Decodificar audio a WAV 16 kHz])
        UC9([Validar requisitos antes de empezar])
        UC10([Implementar y registrar un backend])
        UC11([Ejecutar suite de conformidad])
        UC12([Evaluar un backend contra el corpus])
        UC13([Comparar dos reportes])
        UC14([Ejecutar scripts standalone])
    end

    U --- UC1
    U --- UC6
    U --- UC7
    U --- UC14
    UC1 -. include .-> UC2
    UC1 -. include .-> UC9
    UC1 -. include .-> UC8
    UC3 -. extend .-> UC1
    UC4 -. extend .-> UC3
    UC5 -. extend .-> UC3

    D --- UC10
    D --- UC11
    UC10 -. include .-> UC11

    E --- UC12
    E --- UC13
    UC12 -. include .-> UC2
    UC12 -. include .-> UC8

    UC8 --- FF
    UC2 --- EL
    UC5 --- HF
    UC12 --- TB
```

| Actor | Casos de uso | Punto de entrada |
| ----- | ------------ | ---------------- |
| Usuario | Transcribir, listar backends, revisar resultados, scripts standalone | `python -m audio_transcriber`, `standalone/*.py` |
| Desarrollador | Implementar un `TranscriptionBackend`, pasar la conformidad | `registry.register`, `pytest -k conformance` |
| Evaluador | Medir WER/retención/mezcla de idiomas, comparar configuraciones | `python -m audio_transcriber.evaluation` |

## 4. Diagrama de secuencia — transcribir un lote

Las llamadas entre módulos durante `audio-transcriber -i ./audio -b faster-whisper`. El backend
es intercambiable; las notas indican qué hace cada uno dentro de `transcribe`.

```mermaid
sequenceDiagram
    autonumber
    actor U as Usuario
    participant CLI as cli.main
    participant P as pipeline
    participant R as registry
    participant B as Backend
    participant DS as discovery
    participant A as audio
    participant FF as ffmpeg
    participant C as conformance
    participant FS as Sistema de archivos

    U->>CLI: audio-transcriber -i ./audio -b faster-whisper --option ...
    CLI->>CLI: _parse_option() por cada --option
    CLI->>P: run(input, backend_name, out, glob, config, on_progress)

    P->>R: create(backend_name, **config)
    R->>R: _load_builtins() (import perezoso por extra)
    alt backend desconocido o extra no instalado
        R-->>P: UnknownBackendError / BackendNotInstalledError
        P-->>CLI: ConfigError
        CLI-->>U: error en stderr, exit 3
    end
    R->>B: __init__(**config)
    R-->>P: instancia del backend

    rect rgba(128, 128, 128, 0.1)
        note over P,B: validate(): todo lo verificable antes de empezar
        P->>DS: discover(input, glob)
        DS-->>P: lista de archivos de audio
        P->>A: check_ffmpeg()
        A->>FF: ffmpeg -version
        FF-->>A: versión ≥ 6.0
        P->>B: check_requirements()
        B-->>P: OK (o BackendRequirementError → exit 3)
    end

    P->>FS: prepare_run_directory(out/run-id)
    P->>B: model_version(), effective_config()
    B-->>P: versión y config resuelta
    P->>P: crear Manifest

    loop por cada archivo de entrada
        P->>CLI: on_progress(i, total, archivo)
        CLI-->>U: [i/total] archivo
        P->>FS: InputRecord.from_path() (sha256, tamaño)
        P->>A: decoded(archivo)
        A->>FF: decodificar → WAV mono 16 kHz (temporal)
        FF-->>A: archivo WAV
        A-->>P: ruta del WAV
        P->>B: transcribe(wav, archivo)
        note right of B: faster-whisper: modelo local, VAD por chunk,<br/>pyannote opcional<br/>elevenlabs-scribe: sube el WAV a la API de Scribe
        B-->>P: Transcript normalizado
        P->>A: (fin del contexto) borrar WAV temporal
        P->>C: assert_transcript_conforms(transcript)
        alt transcript válido
            C-->>P: OK
            P->>FS: escribir <stem>.json (array de segmentos)
            P->>P: manifest.inputs += record.with_result()
        else AudioDecodeError / TranscriptionError / ConformanceError
            P->>P: failures += FileFailure, continuar con el siguiente
        end
    end

    P->>FS: escribir manifest.json
    P->>FS: escribir errors.json
    P-->>CLI: RunResult
    CLI-->>U: resumen (ok / fallidos / duración)
    CLI-->>U: exit 0, 1 o 2 según RunResult.status
```

## 5. Diagrama de flujo — evaluación contra el ground truth

El harness de `python -m audio_transcriber.evaluation` transcribe el corpus Bangor Miami
(subconjunto `mixed`: cinco grabaciones en inglés/español, unas 2.5 h) y compara cada hipótesis
con su transcripción de referencia. Con `--talkbank` también puntúa hablantes e idioma por
segmento contra las anotaciones originales de TalkBank. Detalle de las métricas en
[`evaluation.md`](evaluation.md).

```mermaid
flowchart TD
    A(["python -m audio_transcriber.evaluation"]) --> B{"¿--compare BASE CANDIDATO?"}

    %% ── Evaluar una configuración ──
    B -- No --> C{"¿--backend y --label?"}
    C -- No --> Z2(["Error de argumentos, exit 2"])
    C -- Sí --> D["registry.create(backend, **options)"]
    D --> E["audio.check_ffmpeg() + backend.check_requirements()"]
    E -- Falla --> Z3(["AudioTranscriberError, exit 3"])
    E -- OK --> F{"¿mixed.parquet en caché<br/>y válido? (≥100 MB, magic PAR1)"}
    F -- No --> F1["Descargar de Hugging Face"]
    F1 --> F2{"¿Parquet válido?"}
    F2 -- No --> Z3
    F2 -- Sí --> G
    F -- Sí --> G["corpus.load(): Recording por grabación<br/>audio MP3 + transcript de referencia + spanish_share"]
    G --> H["Filtrar con --only, si se dio"]
    H --> I{"¿--talkbank?"}
    I -- Sí --> I1["talkbank.load() por grabación:<br/>enunciados con hablante, tiempos e idioma"]
    I1 -- Archivo ausente --> Z3
    I1 --> J
    I -- No --> J["Crear Report: label, backend, model_version, config"]

    J --> K{"¿Quedan grabaciones?"}
    K -- Sí --> L["audio.decoded(): MP3 → WAV mono 16 kHz"]
    L --> M["backend.transcribe(wav)"]
    L -- Excepción --> M1["result.error = tipo: mensaje"]
    M -- Excepción --> M1
    M1 --> K

    M --> N["Hipótesis: normalize(transcript.text)<br/>minúsculas, sin puntuación ni acentos"]

    subgraph GT ["Ground truth"]
        G1["Referencia con marcado CHAT"] --> G2["clean_reference():<br/>quitar pausas, retracing, continuaciones,<br/>códigos de evento e inline, completar palabras parciales"]
        G2 --> G3["normalize()"]
    end
    G -.-> G1

    N --> S["scoring.score(referencia, hipótesis)"]
    G3 --> S
    S --> S1["jiwer: WER = (S + D + I) / palabras de referencia<br/>retained = (aciertos + sustituciones) / referencia<br/>coverage = palabras hipótesis / referencia<br/>language_mix_error = abs(diferencia de proporción de español)"]

    S1 --> T{"¿Hay referencia TalkBank?"}
    T -- Sí --> T1["language_confusion(): idioma por segmento vs. enunciado,<br/>por tiempo → accuracy, recall por idioma"]
    T1 --> T2{"¿El transcript tiene hablantes<br/>y la referencia es diarizable?"}
    T2 -- Sí --> T3["diarization_error(): DER = missed + false alarm + confusion"]
    T2 -- No --> T4["diarization = null o diarization_skipped"]
    T3 --> U
    T4 --> U
    T -- No --> U["Agregar RecordingResult al Report<br/>segmentos, idiomas detectados, realtime factor"]
    U --> V["Escribir eval/transcripts/grabación.label.txt y .json"]
    V --> K

    K -- No --> W["report.write(eval/label.json)<br/>resumen: medias de WER, retained, coverage, mezcla de idioma"]
    W --> Z0(["Imprimir resumen, exit 0"])

    %% ── Comparar dos reportes ──
    B -- Sí --> X["Cargar ambos reportes"]
    X --> Y{"Por cada grabación puntuada en los dos"}
    Y --> Y1{"¿WER candidato < WER base<br/>y retained candidato ≥ retained base − 0.02?"}
    Y1 -- Sí --> Y2["improved = true"]
    Y1 -- No --> Y3["improved = false<br/>(una mejora de WER perdiendo contenido no cuenta)"]
    Y2 --> Y4["Deltas por grabación y de las medias<br/>WER, retained, coverage, mezcla de idioma, TalkBank"]
    Y3 --> Y4
    Y4 --> Y5{"¿Mejoró en todas las grabaciones?"}
    Y5 -- Sí --> ZC0(["JSON de comparación, exit 0"])
    Y5 -- No --> ZC1(["JSON de comparación, exit 1"])
```
