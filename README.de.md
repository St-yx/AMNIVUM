<p align="right">
  <a href="README.md">🇬🇧 English</a> |
  <strong>🇩🇪 Deutsch</strong>
</p>

# VIOLET — **V**irtual **I**ntelligence **O**bserving to **L**earn, **E**volving to **T**hink

**Violet** ist ein lokal gehosteter "KI-Companion" mit persistentem Gedächtnis, dynamischem Gefühlszustand und einer modularen Architektur, die auf Wachstum über Zeit ausgelegt ist. Das Projekt trennt Kernfunktionen radikal voneinander: Das Sprachmodell selbst ist nur die Stimme. Alles andere ist darum herum gebaut — in Anlehnung an das menschliche Gehirn.

---

## Architekturüberblick

AMNIVUM ist der technische Stack, der Violet "erzeugt". Er besteht aus unabhängigen Modulen, die asynchron über eine zentrale Message-Queue kommunizieren. Kein Modul ruft ein anderes direkt auf — sie publizieren und abonnieren, was Latenz vorhersagbar hält und das System erweiterbar macht.

```
                       ┌───────────────┐
                       │     SENSUS    │  Wahrnehmungsschicht (Audio, Video, STT)
                       └───────┬───────┘
                               │
                       ┌───────▼───────┐
                       │     KORTEX    │  Prompt-Assembler — LLM-seitiges Modul
                       └───┬──────┬────┘
                           │      │
              ┌────────────▼─┐  ┌─▼────────────┐
              │   MEMORIA    │  │   INGENIUM   │  Gedächtnis und Emotionen laufen parallel pro Turn
              └────────────┬─┘  └─┬────────────┘
                           │      │
                       ┌───▼──────▼───┐
                       │     LLM      │  Sprachzentrum
                       └──────────────┘
```

Jedes Modul ist ein Python-asyncio-Task. Die gemeinsame Message-Queue ist der einzige Kommunikationskanal.

*Das Projekt befindet sich noch in intensiver Entwicklung. Die Architektur ist größtenteils geplant und spezifiziert. Die Implementierung hat für einige Module begonnen. Änderungen können und werden eintreten, während das Projekt Form annimmt.*

**Aktueller Code: 796 Zeilen Python**

---

## Module

### KORTEX — Prompt-Assembler

KORTEX orchestriert den Turn-Zyklus. Er empfängt Klartext vom User über SENSUS, verteilt ihn an MEMORIA und INGENIUM, sammelt deren Ausgaben und baut den finalen Prompt für das Sprachmodell zusammen. Auf der Ausgangsseite entpackt er die LLM-Antwort und leitet sie symmetrisch durch dieselbe Pipeline zurück.

KORTEX interpretiert nicht — er baut nur. Sämtlicher semantische und emotionale Kontext wird von anderen Modulen geliefert.

**Status:** Architektur zu 99% spezifiziert. Implementierung bei 25% (Input-Handler experimentell).

---

### MEMORIA — Erinnerungssystem

MEMORIA verwaltet alles, woran sich Violet erinnert. Es arbeitet auf drei Schichten mit unterschiedlichen Lebensdauern und führt pro Turn eine Chunking- und Retrieval-Pipeline aus.

#### Speicherschichten

**MEMORIA-LONG** — persistenter Vektorspeicher (Qdrant Collection 1)  
Konsolidiertes Langzeitwissen über den User, Violet selbst und die Welt. Wird ausschließlich während der Offline-Konsolidierung beschrieben. Einträge mit `core: true` sind decay-immun. Alle anderen folgen einer konfigurierbaren Decay-Funktion. Das Retrieval erfolgt thematisch per Cosine-Similarity über bis zu drei Hauptthemen.

**MEMORIA-MID** — session-übergreifender Zwischenspeicher (Qdrant Collection 2)  
Chunks aus der laufenden Session, inklusive Emotions-Tags die von INGENIUM vergeben wurden. Kein Decay — wird bei der Konsolidierung vollständig geleert. Liefert aktuellen Gesprächskontext und Material zur Widerspruchserkennung.

**MEMORIA-SHORT** — In-Memory-Kontext-Buffer (nur RAM)  
Das aktive Kontextfenster für KORTEX. Hält ca. 20 Chunks, die nach jedem Turn dynamisch neu ausgewählt und bewertet werden. Wird nie persistiert. Strukturiert als gewichtete Relevanz-Queue aus LONG und MID, mit garantierter Repräsentation aller Wissensarten (User-Wissen, Weltwissen, KI-Selbstwissen).

#### Buffer-Slot-Logik

Der Buffer füllt sich über eine zweistufige "Schablone". Pass 1 garantiert Mindestrepräsentation pro Wissensquelle (User, Welt, KI, MID) und verteilt ungenutzte Kapazität nach unten weiter. Pass 2 füllt verbleibende Slots dynamisch per Round-Robin über verfügbare Pools. Themen 2 und 3 (Satelliten-Cluster) erhalten eine feste Seitenallokation. Das Ergebnis ist ein Buffer der immer so voll ist wie vorhandene Daten erlauben, ohne eine einzelne Quelle künstlich aufzublähen.

#### Chunking-Pipeline

Eingabetext wird an Satzgrenzen per Regex gesplittet, mit einem mehrsprachigen Sentence-Transformer eingebettet, dann semantisch zusammengeführt: benachbarte Sätze über einem Cosine-Similarity-Schwellwert werden zusammengefügt, solange die kombinierte Wortanzahl im Rahmen bleibt. Zu kurze Restchunks werden an Nachbarn angehängt statt verworfen. Das Merge-Embedding ist nur eine Annäherung — das finale Qdrant-Embedding wird frisch aus dem zusammengeführten Text berechnet.

#### Retrieval

Pro Turn werden die Eingabe-Chunks per agglomerativem Clustering auf ihren Embeddings nach Themen gruppiert. Pro Thema (bis zu drei) wird ein gewichteter Durchschnittsvektor berechnet. Jeder Topic-Vektor wird gegen den Cluster-Graph abgeglichen um den passenden LONG-Cluster zu finden. Das Retrieval führt dann drei parallele Queries pro Cluster durch — eine pro Wissensquelle — um Repräsentation zu garantieren, gefolgt von einem breiteren Similarity-Sweep für verbleibende Kandidaten. MID wird separat nach Similarity und Aktualität abgefragt.

Der Cluster-Graph ist ein JSON-Index aller LONG-Cluster-Centroids und ihrer Nachbarschaftsbeziehungen, der offline vom Konsolidierer gepflegt wird. Er ermöglicht Satelliten-Cluster-Auswahl zur Laufzeit ohne zusätzliche Qdrant-Queries.

#### Importance-Gate

Bevor ein Turn-Chunk in MID geschrieben wird, werden drei Bedingungen gleichzeitig geprüft: der Chunk ist zu kurz, er ist zu ähnlich zu bestehenden Cluster-Centroids (redundant), und sein Emotions-Signal ist flach (hoher Neutral-Score, niedrige Amplitude). Treffen alle drei zu, wird der Chunk nicht in MID geschrieben. Er landet dennoch immer im Session-Log — der Offline-Konsolidierer sieht alles.

#### Themenwechsel-Erkennung

Nach jedem Turn wird das Turn-Embedding gegen einen Sliding-Window-Durchschnitt der letzten Turn-Embeddings verglichen. Fällt die Similarity unter einen Schwellwert, wird ein Themenwechsel erkannt und der Buffer mit frischen Qdrant-Queries gegen den neuen Topic-Vektor neu befüllt.

#### Offline-Prozesse

**Konsolidierer** — läuft während LLM-Idle-Zeit, analog zum menschlichen REM-Schlaf. Liest das Session-Log, chunked und embeddet es, übergibt es an INGENIUM-OFFLINE zur Importance-Bewertung und Widerspruchserkennung, schreibt dann selektiv in MEMORIA-LONG. Leert MEMORIA-MID vollständig nach Abschluss.

**Eraser** — läuft periodisch. Wendet die Decay-Funktion auf LONG-Einträge an und entfernt unter dem Importance-Schwellwert liegende. Core-Einträge bleiben unangetastet.

**Status:** Architektur zu 99% spezifiziert. Implementierung zu 50% (Online-Pipeline — Chunking, Retrieval, Buffer, Importance-Gate, Themenwechsel) experimentell. Offline-Prozesse (Konsolidierer, Eraser) spezifiziert.

---

### INGENIUM — Persönlichkeit und Affect

INGENIUM gibt Violet einen persistenten Gefühlszustand, der jede Antwort beeinflusst ohne in irgendeinen Prompt fest eingebaut zu sein.

**INGENIUM-STATIC** — ein Charakter-LoRA das in das Basismodell gemergt wird. Definiert Eigenschaften, Temperament, Reaktionsmuster und Prompt-Format. Zur Laufzeit nicht änderbar — erfordert Retraining.

**INGENIUM-INTERPRETER** — ein mehrsprachiger Emotions-Klassifikator (XLM-RoBERTa-basiert, 11 Emotions-Dimensionen). Klassifiziert jeden Eingabe-Chunk und jeden LLM-Output-Chunk symmetrisch und erzeugt eine Wahrscheinlichkeitsverteilung über alle 11 Labels pro Chunk.

**INGENIUM-AFFECT** — eine persistente JSON-Zustandsdatei mit `global_affect`, ein langsam veränderlicher emotionaler Grundton, über alle Turns akkumuliert

Der Affect-Zustand wird pro Turn in zwei Durchläufen aktualisiert: einmal vor dem Prompt-Bau (mit abgerufenen Clean-Tags), einmal nachdem KORTEX das vollständige Bild hat (mit Turn-Tags, Raw-Tags und Acceptance-Tags gewichtet zusammen).

Die Drift-Erkennung vergleicht eingehende Emotions-Vektoren gegen die Cluster-Historie. Geringer Drift validiert den bestehenden Affect. Hoher Drift setzt ein Conflict-Flag an KORTEX, das eine Rückfrage-Sequenz des LLM auslösen kann.

**Status:** Architektur zu 99% spezifiziert. Implementierung zu 25% (Interpreter experimentell).

---

### SENSUS — Wahrnehmungsschicht

SENSUS verwaltet alle peripheren Ein- und Ausgaben. Es ist in drei Ebenen gegliedert: CLIENT (Mikrofon, Kamera, Lautsprecher), NODE (STT via Whisper, Avatar-Rendering, Sprachsynthese) und HOST (Metadaten-Interpretation, KORTEX-Schnittstelle). Audio- und Videostreams fließen als strukturierte Textbeschreibungen einwärts, KORTEX-Ausgaben fließen als gerendertes Audio und Avatar-Bewegung auswärts. Nach Fertigstellung wird es verschiedene Interaktionsmodi geben — nur Chat, Sprache, und Video+Sprache — je nach verfügbarer Hardware und Systemleistung.

**Status:** Architektur geplant. Implementierung nicht begonnen.

---

### ANIMUS — Innere Stimme

ANIMUS ist Violets autonomes Innenleben — ein zweiter LLM-Prozess der außerhalb von User-Turns im eigenen Takt läuft. Er hat vollen Zugriff auf MEMORIA und INGENIUM und kommuniziert zurück in den Hauptstack auf dieselbe Weise wie KORTEX.

Die aktive Hälfte von ANIMUS produziert Ausgaben: unaufgeforderte Nachrichten, Fragen, Reaktionen auf SENSUS-Umgebungsinput und Signale die Violets Affect-Zustand ohne jede User-Aktion verschieben. Die passive Hälfte, die über viel längere Zeiträume entwickelt wird, soll schrittweise deterministische Entscheidungen in MEMORIA und INGENIUM durch trainierte Intuition ersetzen — was zu erinnern ist und wie sich etwas anfühlt — aufgebaut aus den Interaktionsdaten die NUCLEUS natürlich über tausende von Turns produziert.

**Status:** Architektur noch in Planung. Implementierung beginnt nach Stabilisierung des Core-Stacks.

---

## Sprachmodell

Das LLM ist nicht Violets Gehirn — es ist ihre Stimme. Kontext wird vollständig durch MEMORIAs Buffer getragen. Emotionale Einordnung liefert ausschließlich INGENIUM. KORTEX assembliert einen strukturierten Prompt der Affect-Tags, chunk-level Emotions-Metadaten und Flags für Konflikte oder Wissenslücken enthält. Das Modell soll aus dem sprechen was es bekommt, nicht aus dem was es selbst weiß.

Ein LoRA-Adapter trainiert das Modell darauf, dieses Prompt-Format als natürlich zu behandeln, bei fehlendem Kontext nachzufragen statt selbst zu interpretieren, und eine konsistente Charakterstimme beizubehalten. Das Basismodell wird für qualitativ hochwertige (deutsche) Sprachausgabe, Instruction-Following-Fähigkeit und Finetuning-Eignung ausgewählt — im 3–7B Parameter-Bereich, uncensored.

**Status:** Modellauswahl und -tests laufen. LoRA-Trainingsdaten-Strategie definiert. Implementierung nicht begonnen.

---

## Tech Stack

| Komponente | Technologie | Hardware |
|---|---|---|
| Laufzeit | Python asyncio | CPU |
| Vektorspeicher | Qdrant (2 Collections) | CPU / RAM |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 | CPU |
| Emotions-Klassifikator | multilingual-emotion-classification (XLM-RoBERTa) | CPU |
| LLM-Backend | llama.cpp oder Ollama | GPU |
| Infrastruktur | Docker Compose | — |

---

## Projektstruktur

```
AMNIVUM/
├── nucleus/                  # Haupt-Python-Package
│   ├── kortex/               # Prompt-Assembler, Input/Output-Handling
│   ├── memoria/              # Erinnerungssystem (core, retriever, short)
│   ├── ingenium/             # Affect-Zustand, Emotions-Klassifikator
│   ├── sensus/               # Wahrnehmungsschicht
│   ├── animus/               # Innere Stimme
│   └── shared/               # Messages, Queues, Services, Config
├── data/                     # Persistente Laufzeitdaten (nicht versioniert)
│   ├── affect.json
│   ├── cluster_graph.json
│   └── session_log/
├── vector_storage/           # Qdrant-Datenverzeichnis (nicht versioniert)
├── docker-compose.yml        # Infrastrukturdefinition
├── .env                      # Alle Konfigurationen und Secrets (nicht versioniert)
└── pyproject.toml
```

---

## Offene Punkte

- [ ] Qdrant-Collection-Schema und Metadaten-Felder finalisieren
- [ ] Alle Similarity-Schwellwerte empirisch gegen echte Daten kalibrieren
- [ ] INGENIUM-Affect-Update in den Turn-Zyklus einbinden
- [ ] Offline-Konsolidierer und Eraser implementieren