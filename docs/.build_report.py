from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path

ROOT = Path(r"C:\Users\vinay\OneDrive\Documents\Finance\Test Cases\NLP\MML_Project\CineComic_MML")
OUT = ROOT / "docs" / "CineComic_MML_Final_Year_Project_Report.docx"
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin = Inches(.72), Inches(.72)
sec.left_margin, sec.right_margin = Inches(.82), Inches(.82)
sec.header_distance = sec.footer_distance = Inches(.34)
sec.different_first_page_header_footer = True

INK, NAVY, TEAL, GREY = "171717", "17324D", "2C7180", "52616B"
normal = doc.styles["Normal"]
normal.font.name = "Arial"; normal.font.size = Pt(10.5); normal.font.color.rgb = RGBColor.from_string(INK)
normal._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
normal.paragraph_format.space_after = Pt(6); normal.paragraph_format.line_spacing = 1.08
for name, size in (("Title",27),("Heading 1",19),("Heading 2",13)):
    s=doc.styles[name]; s.font.name="Arial"; s.font.size=Pt(size); s.font.bold=True
    s.font.color.rgb=RGBColor.from_string(INK); s._element.rPr.rFonts.set(qn("w:ascii"),"Arial")
    s._element.rPr.rFonts.set(qn("w:hAnsi"),"Arial")
    s.paragraph_format.keep_with_next=True; s.paragraph_format.space_after=Pt(6)
hp=sec.header.paragraphs[0]; hp.alignment=WD_ALIGN_PARAGRAPH.RIGHT
r=hp.add_run("CINECOMIC-MML  |  FINAL YEAR PROJECT REPORT"); r.font.size=Pt(8); r.font.color.rgb=RGBColor.from_string(GREY)
fp=sec.footer.paragraphs[0]; fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=fp.add_run("CineComic-MML  •  "); r.font.size=Pt(8); r.font.color.rgb=RGBColor.from_string(GREY)
f=OxmlElement("w:fldSimple"); f.set(qn("w:instr"),"PAGE"); fp._p.append(f)
sec.first_page_footer.paragraphs[0].text=""

def table(headers, rows, widths=None, font=9):
    t=doc.add_table(rows=1, cols=len(headers)); t.style="Table Grid"; t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
    for i,x in enumerate(headers): t.rows[0].cells[i].text=str(x)
    for row in rows:
        cells=t.add_row().cells
        for i,x in enumerate(row): cells[i].text=str(x)
    for ri,row in enumerate(t.rows):
        for ci,c in enumerate(row.cells):
            c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if widths: c.width=Inches(widths[ci])
            tcpr=c._tc.get_or_add_tcPr(); sh=OxmlElement("w:shd"); sh.set(qn("w:fill"), NAVY if ri==0 else ("F4F6F7" if ri%2==0 else "FFFFFF")); tcpr.append(sh)
            for p in c.paragraphs:
                p.paragraph_format.space_after=Pt(2); p.paragraph_format.space_before=Pt(2); p.paragraph_format.line_spacing=1
                for run in p.runs:
                    run.font.name="Arial"; run.font.size=Pt(font)
                    if ri==0: run.font.bold=True; run.font.color.rgb=RGBColor.from_string("FFFFFF")
    doc.add_paragraph().paragraph_format.space_after=Pt(1)

def page(kicker,title,body,bullets=None,tab=None,img=None,note=None,first=False):
    if not first: doc.add_page_break()
    p=doc.add_paragraph(); p.paragraph_format.space_after=Pt(2)
    r=p.add_run(kicker.upper()); r.font.size=Pt(8.5); r.font.bold=True; r.font.color.rgb=RGBColor.from_string(TEAL)
    doc.add_paragraph(title,style="Heading 1")
    for para in body:
        p=doc.add_paragraph(para); p.paragraph_format.widow_control=True
    if bullets:
        for b in bullets:
            p=doc.add_paragraph(style="List Bullet"); p.paragraph_format.space_after=Pt(3); p.add_run(b)
    if tab: table(*tab)
    if img:
        p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(3)
        p.add_run().add_picture(str(img),width=Inches(5.35))
    if note:
        p=doc.add_paragraph(note); p.paragraph_format.space_after=Pt(1)
        for r in p.runs: r.italic=True; r.font.size=Pt(8.5); r.font.color.rgb=RGBColor.from_string(GREY)

def phase(n,title,purpose,method,design,result,caveat,artifact):
    page(f"Chapter 5 | Phase {n}",title,[
        "Purpose. "+purpose,
        "Method. "+method,
        "Implementation. "+design,
        "Observed result. "+result,
        "Interpretation. "+caveat,
        "Traceability. "+artifact
    ])

# 1 Cover
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(54)
r=p.add_run("FINAL YEAR PROJECT REPORT"); r.bold=True; r.font.size=Pt(14); r.font.color.rgb=RGBColor.from_string(TEAL)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(36)
r=p.add_run("MULTIMODAL VIDEO\nTO COMIC GENERATION SYSTEM"); r.bold=True; r.font.size=Pt(27)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(12)
r=p.add_run("CineComic-MML"); r.font.size=Pt(18); r.font.color.rgb=RGBColor.from_string(NAVY)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(24)
p.add_run("Phase-wise methods, implementation, and evaluation\n").font.size=Pt(12)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(47)
p.add_run("Submitted by\n").bold=True; p.add_run("[Student Name]  |  [Register Number]\n\n")
p.add_run("Department of [Department Name]\n[Institution Name]\n\n")
p.add_run("Project Supervisor: [Supervisor Name]\n\nAcademic Year: [Year]")
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(42)
r=p.add_run("Based on the implemented project and recorded evaluation job 876c14763e77"); r.italic=True; r.font.size=Pt(9); r.font.color.rgb=RGBColor.from_string(GREY)

# 2
page("Front matter","Certificate",[
"This is to certify that the project report entitled “Multimodal Video to Comic Generation System” is a record of the project work carried out by [Student Name], [Register Number], under the supervision of [Supervisor Name], in partial fulfilment of the requirements for the award of [Degree / Programme] of [Institution Name].",
"The report describes the existing CineComic-MML implementation and a recorded evaluation run. It documents the system methods, output artifacts, validation results, and limitations available at the time of preparation. Approval wording should be aligned with the institution’s prescribed format."
],tab=(["Role","Name","Signature","Date"],[
["Project Supervisor","[Supervisor Name]","________________","__________"],
["Head of Department","[HOD Name]","________________","__________"],
["External Examiner","[Examiner Name]","________________","__________"]],[1.5,1.7,1.7,1.15]))

# 3
page("Front matter","Student Declaration",[
"I, [Student Name], bearing Register Number [Register Number], declare that this report titled “Multimodal Video to Comic Generation System” has been prepared by me as part of the requirements for [Degree / Programme] at [Institution Name].",
"The implementation and results are attributed to the existing project repository and its recorded artifacts. External research is cited in the references. The report distinguishes internal structural checks from accuracy measurements that could not be computed without labelled reference data.",
"I understand that the final submission must follow the academic integrity and formatting rules of my institution. Any additional originality statement, signature, or mandated wording should be inserted in the approved form."
],tab=(["Place","Date","Student signature"],[["[Place]","[Date]","________________"]],[2,2,2.1]))

# 4
page("Front matter","Acknowledgement",[
"This editable page should be finalized with acknowledgements that reflect actual contributions.",
"I express my sincere gratitude to [Supervisor Name] for guidance and feedback during this project. I thank the faculty members of [Department Name], [Institution Name], for academic support and access to facilities used in this work.",
"I acknowledge the open-source communities and researchers whose software and publications informed the project’s methods in speech recognition, diarization, visual-language representation, face analysis, active-speaker detection, emotion recognition, and comic layout.",
"Finally, I thank [individuals or organizations, if applicable] for their assistance. Include only contributors and support that can be verified."
])

# 5
page("Front matter","Abstract",[
"CineComic-MML is a multimodal system that converts video into a sequence of comic pages. Its staged pipeline combines media validation, speech recognition, speaker diarization, shot/frame selection, face detection and tracking, visual-text alignment, active-speaker matching, emotion estimation, panel layout, image stylization, and dialogue-aware rendering. A local Flask application supports upload, job monitoring, cancellation, preview, and export.",
"This report studies the recorded job 876c14763e77, a 261.545-second workshop-scene clip with 1280 × 544 resolution and 23.976 frames per second. The completed run produced 110 transcript segments, 74 diarized turns, 49 shots, 355 retained candidate frames, 126 face tracks, 79 selected keyframes, 8 scene summaries, 79 panel summaries, 14 comic pages, and 89 rendered bubbles. All 110 transcript segments were placed. Structural validation found no missing or duplicated segments, no transcript text mismatch, and no measured bubble overlap or out-of-panel placement.",
"The evidence also shows unresolved quality problems. Only 15 of 74 turns were provisionally matched to visible tracks, no persistent speaker-to-character mapping was accepted, five frame IDs repeated, and validation status was review_required. Ground-truth transcript, diarization, identity, and emotion labels were unavailable, so this report does not claim accuracy for these tasks. The project demonstrates an integrated generation workflow and useful consistency checks; character attribution and content-level quality require further evaluation."
],bullets=["Keywords: multimodal video processing; ASR; diarization; face tracking; active speaker; comic generation; scene summaries"])

# 6
page("Front matter","Contents",[
"Front matter: Certificate; Student Declaration; Acknowledgement; Abstract; Contents; Figures, Tables, and Abbreviations.",
"Chapter 1 — Introduction; problem definition, objectives, and scope.",
"Chapter 2 — Background and related work.",
"Chapter 3 — Requirements and constraints.",
"Chapter 4 — System architecture and data flow.",
"Chapter 5 — Phase-wise methodology: ingestion; ASR; diarization; shot/frame selection; face tracking; cross-modal alignment; active speaker matching; keyframe/story planning; emotion; layout; styling; assembly/validation.",
"Chapter 6 — Data contracts, conversation order, and scene summaries.",
"Chapter 7 — Application and implementation.",
"Chapter 8 — Experimental setup and evaluation protocol.",
"Chapter 9 — Quantitative results and qualitative output review.",
"Chapter 10 — Verification, limitations, and discussion.",
"Chapter 11 — Conclusion and future work.",
"Research references; project artifacts; appendices for reproduction and detailed results.",
"Page numbers are provided by the word processor. Institution-specific additions may shift pagination."
])

# 7
page("Front matter","Figures, Tables, and Abbreviations",[
"Figures: Figure 9.1, sample generated comic page; Figure 9.2, structural coverage and speaker evidence summary.",
"Tables: Table 3.1, functional requirements; Table 8.1, evaluation setup; Table 9.1, phase-wise results; Table 10.1, output validation.",
"Abbreviations: ASR, Automatic Speech Recognition; A/V, audio/video; CBZ, Comic Book ZIP archive; CFR, Constant Frame Rate; CLIP, Contrastive Language–Image Pre-training; CPU, Central Processing Unit; DER, Diarization Error Rate; FPS, Frames per Second; IoU, Intersection over Union; MAE, Mean Absolute Error; MML, Multimodal Learning; ROI, Region of Interest; RTF, Real-Time Factor; VAD, Voice Activity Detection; WER, Word Error Rate."
])

# 8
page("Chapter 1","Introduction",[
"Video combines images, motion, speech, sound, and scene changes. Converting it into a comic is therefore more than sampling frames and placing subtitles over them. A useful adaptation must select representative moments, preserve the timeline, arrange panels in a readable sequence, retain dialogue, and communicate who is speaking when the evidence supports the decision.",
"CineComic-MML uses a staged pipeline to perform these tasks. It validates media, recognizes speech, estimates speaker turns, detects shots and faces, aligns dialogue with visual moments, selects keyframes, estimates emotion, arranges pages, stylizes panels, and renders dialogue balloons and captions. Intermediate JSON and metrics make outputs inspectable. The web application provides an interface for submitting a clip and reviewing generated pages.",
"The project is intended as an assistive storyboard and visual-summary tool. Its output can save manual preparation effort, but generation is not a substitute for editorial review. Errors in recognition, identity, scene interpretation, or frame choice can propagate into later stages. The evaluation in this report therefore separates operational completion from semantic accuracy."
])

# 9
page("Chapter 1","Problem Definition, Objectives, and Scope",[
"Problem. The system must convert a time-based audiovisual signal into a page-based narrative without losing timing or context. It must decide what was said, who said it, which visible face corresponds to the voice, which frame depicts the moment, how many panels are needed, and where a bubble can be placed without covering the scene.",
"Objectives: (1) validate and normalize media; (2) generate timestamped dialogue and anonymous speaker turns; (3) find shots, faces, tracks, and cross-modal correspondences; (4) select chronological panel moments; (5) estimate emotion and scene context; (6) lay out and stylize panels; (7) place dialogue without accidental omission or duplication; (8) validate exports and provide a review workflow.",
"Scope. This report describes the existing twelve-stage implementation and one completed evaluation run. It does not claim generalization across genres or languages. There is no human-labelled reference set for the current clip, so WER, DER, identity accuracy, frame-selection precision, emotion accuracy, and summary faithfulness are not reported.",
"The project succeeds operationally if artifacts are generated and structural invariants hold. Content success requires correct words, identities, actions, and summaries. Current evidence is stronger for the first category."
],bullets=["Research question: how can the existing workflow preserve visual and dialogue context through comic rendering?","Evaluation question: what passed on the recorded clip, and what still requires human review?"])

# 10
page("Chapter 2","Background and Related Work: Speech and Alignment",[
"Automatic speech recognition converts audio into text and may estimate word or segment timestamps. Whisper was trained on large weakly supervised multilingual data and reports robustness across varied audio [1]. This project uses faster-whisper, a small English model, VAD, and configurable segmentation/filtering. The paper motivates the model family; it does not establish the error rate of this run.",
"Speaker diarization estimates when anonymous voices speak. The configured pyannote.audio pipeline segments and clusters speaker activity [2]. Labels such as SPEAKER_00 are cluster identifiers inside a recording, not character names. DER is normally computed against reference speaker activity. This project run has no such reference.",
"CLIP-style visual-language encoders map images and text into a shared embedding space for similarity-based retrieval [3]. CineComic-MML uses OpenCLIP ViT-B-32 to rank frame/dialogue candidates. A high similarity score is not proof that an image shows the described action or speaker. Camera cuts, lighting, figurative dialogue, and screen inserts can make a semantically related frame narratively wrong.",
"Accordingly, this report treats ASR segment counts and alignment artifacts as evidence of pipeline behavior, not accuracy."
])

# 11
page("Chapter 2","Background and Related Work: Vision and Comics",[
"Face-analysis systems use embeddings to compare detected faces. ArcFace is a margin-based face representation approach [4]. The project uses InsightFace buffalo_l detections and embeddings, temporal track association, and identity clustering. These outputs are internal visual hypotheses; they do not provide verified identities and can fragment a person across views.",
"Active speaker detection asks which visible person is speaking at a time. Active Speakers in Context evaluates audiovisual cues and contextual reasoning [5]. Phase 7 here is a project-specific heuristic combining timing, track confidence, mouth-region movement, and an upper-face control. It does not implement the published model. The current low match rate confirms this is an open weakness.",
"SUPERB proposes a benchmark for comparing speech-processing systems [6]. CineComic-MML uses separate text, audio, and face emotion models and fuses their available scores. Inference from any single modality can be wrong, particularly for sarcasm, short utterances, poor audio, or an image of a listener.",
"Comic-generation work has studied multi-page layouts and emotion-aware balloons [7]. This project uses layout heuristics and configurable balloon geometry. These references provide context; the project’s own artifacts and visual review supply the evidence for this report."
])

# 12
page("Chapter 3","Requirements and Constraints",[
"Functional requirements are video upload and validation; phase orchestration; durable intermediate artifacts; timestamped transcript and visual references; dialogue-preserving page assembly; result preview; cancellation; and export. Each panel should retain its timestamp and frame identity, while uncertain speaker identity should remain unknown rather than be assigned to whoever is visible.",
"Quality requirements include reproducibility from saved YAML and seeds where supported; explicit segment IDs; detection of missing or duplicated dialogue; valid frame references; summary presence; bubble geometry checks; and reviewable model uncertainty.",
"Operational constraints include model initialization and download time, CPU-heavy face analysis, gated pyannote access, FFmpeg/FFprobe availability, and possible partial artifacts after cancellation. The app accepts videos up to 512 MB; ingestion configuration caps duration at 300 seconds. The evaluation clip is within that limit.",
"Research constraint is the lack of annotated test data. One unlabelled example is sufficient for exercising all phases but cannot establish model accuracy, user comprehension, or generalization."
],tab=(["Requirement","Evidence"],[
["Media timing","Frame counts agree; A/V duration gap 0.042 s"],
["Dialogue retention","110 of 110 segments placed; no duplicates detected"],
["Uncertainty handling","59 of 74 speaker turns retained as unknown"],
["Panel geometry","No measured bubble overlap or overflow"],
["Context fields","8 scene and 79 panel summaries present"],
["Human review","Web UI exposes preview, stage detail, and validation report"]
],[2.0,4.2]))

# 13
page("Chapter 4","System Architecture and Data Flow",[
"The architecture separates media preprocessing, speech analysis, visual analysis, cross-modal selection, and rendering. The CLI runner invokes twelve stages in fixed order. YAML fragments hold model choices and thresholds. Intermediate files are placed in data/ and results/; final outputs live in output/.",
"Speech and visual evidence meet in alignment and active-speaker matching. The renderer consumes transcript, keyframe, layout, speaker, emotion, and story-plan records. Flask starts main.py as a child process and exposes upload, progress, cancellation, output preview, and downloads to a standalone HTML/CSS/JavaScript client."
],tab=(["Stage group","Dependencies and outputs"],[
["Input","Upload → ingestion → normalized video, audio, metadata"],
["Speech","Audio → ASR transcript → diarized anonymous turns"],
["Vision","Frames/shots → face detections and tracks"],
["Fusion","Transcript + frames + tracks → alignment, speaker evidence, keyframes"],
["Presentation","Keyframes + dialogue + summaries → layout, styling, pages"],
["Review","Validation JSON, PNG pages, PDF, CBZ, HTML"]
],[1.35,4.85]),note="Figure 4.1. Simplified dataflow. Stages also read shared metadata and configuration.")

# 14–25 phases
phase(1,"Media Ingestion and Validation",
"Establish a decodable and time-consistent input before inference. The stage probes media metadata, compares audio and video duration, checks seeking and decoding, and extracts mono 16 kHz audio.",
"FFprobe and OpenCV/FFmpeg-based utilities inspect the source. If required, a working copy can be normalized to CFR. Time is measured in seconds from the start; the frame index is computed as round(t × FPS).",
"Configured gates require frame-count difference ≤1%, A/V duration difference ≤0.1 s, zero decode errors, and 20 valid random-frame reads. The clip is H.264, 1280 × 544, 24000/1001 FPS, 6271 frames, with audio at 16 kHz.",
"All gates passed: zero frame-count difference, 0.042-second A/V gap, zero decode errors, and 20/20 readable frames. Runtime was 12.191 seconds.",
"This validates media integrity, not story relevance or model correctness. A readable frame can still be the wrong panel.",
"Artifacts: data/metadata/876c14763e77.json and .metrics.json; config: configs/phase1.yaml.")

phase(2,"Automatic Speech Recognition and Segmentation",
"Create timestamped text units for alignment and balloon rendering.",
"The run uses faster-whisper small, English, CPU int8, VAD filtering, and beam size 5. Configurable filters use no-speech and log-probability thresholds, remove known generic hallucination phrases, split long units, and merge short fragments.",
"Records carry stable dialogue IDs, text, start/end times, and optional speaker IDs. Timestamps are estimates and can shift with overlap, noise, music, or rapid turn changes.",
"The run emitted 110 segments, dropped none, found no structural timeline problems, and averaged 6.13 words per segment. RTF was 0.5094; runtime 133.259 seconds.",
"No reference transcript or manual timestamps existed. WER, start-time MAE, and hallucination rate were not evaluated. Structural pass means downstream phases could read the artifact, not that every word is correct.",
"Artifacts: data/subtitles/876c14763e77.json and .metrics.json; config: configs/phase2.yaml.")

phase(3,"Speaker Diarization",
"Partition speech into anonymous speaker turns and align speaker IDs with transcript intervals.",
"The configured pyannote/speaker-diarization-3.1 pipeline estimates voice activity and clusters acoustically similar regions. Merge-gap and minimum-turn thresholds filter intervals. JSON and RTTM retain speaker IDs and timing.",
"A cluster ID is not a real character name. Short utterances, overlapping speakers, noise, and dramatic audio can fragment or merge voices.",
"The artifact contains 74 turns and labels all 110 transcript records with three anonymous IDs. No structural problems were recorded; runtime was 491.677 seconds.",
"No reference speaker annotation was provided, so DER and attribution accuracy are unavailable. Any diarization error can propagate to visual matching and bubbles.",
"Artifacts: data/diarization/876c14763e77.raw.json, .rttm, .metrics.json; config: configs/phase3.yaml.")

phase(4,"Shot Detection and Candidate Frame Pool",
"Find candidate moments that can support dialogue alignment and story selection.",
"A content-change detector estimates shot boundaries. Frames are sampled near speech windows and filtered for duplicates, darkness, blur, and borders. Settings include a 0.5 s minimum scene, 2 FPS nominal sampling, 0.75 s padding, and 3–6 candidates per segment.",
"Backfill can add visual candidates when a speech interval is sparse. These frames are a pool for later ranking, not final panels.",
"The run found 49 shots and kept 355 of 637 sampled frames. It removed 114 duplicate, 41 dark, and 127 blurry frames, with 116 backfilled. Reported dialogue coverage was 0.991; runtime was 73.14 s.",
"Operational gates passed, but there is no annotated boundary reference. The eventual opening scenic panel shows why threshold success does not prove narrative relevance.",
"Artifacts: data/shots/876c14763e77.json and .metrics.json; contact sheet in results/phase4; config: configs/phase4.yaml.")

phase(5,"Face Detection, Tracking, and Identity Clustering",
"Create visual observations and temporal tracks to support matching voice turns with visible people.",
"InsightFace buffalo_l runs on CPU, sampling near 1 FPS. Detections are filtered by size and score, then associated using spatial overlap and face-embedding similarity. Clustering gives internal identity groups; uncertain detections remain unknown.",
"Temporal tracks reduce reliance on one still image, but missed detections, cuts, occlusion, and profile views can fragment tracks. Track IDs do not identify real people.",
"The run read 262/262 sampled frames, made 266 detections, and built 126 tracks: 49 known and 77 unknown. Processing took 2106.657 s (7.204 s average per sampled frame).",
"No identity labels were available. Counts are not accuracy. CPU processing is expensive and many tracks remain unknown, limiting downstream speaker matching.",
"Artifacts: data/faces and data/tracks for job 876c14763e77; results/phase5 summary; config: configs/phase5.yaml.")

phase(6,"Cross-Modal Dialogue and Frame Alignment",
"Rank visual frames that may correspond to dialogue or visual action cues.",
"OpenCLIP ViT-B-32 with laion2b_s34b_b79k creates 512-dimensional image and text embeddings. Similarity ranks candidate frame/text pairs; timestamp proximity and available shot data constrain choices. Similarity is retrieval evidence, not a scene-truth classifier.",
"Alignment records connect dialogue IDs with candidate frames and scores. The selected frame should remain temporally close to the utterance.",
"There are 110 transcript alignment records and 303 mapped frame-embedding rows. The later story phase reports 185 frames with semantic scores.",
"No human relevance labels exist, so precision, recall, and ranking quality were not calculated. Semantically related frames can still show the wrong speaker or action.",
"Artifact: data/alignment/876c14763e77.json and data/embeddings; config: configs/phase6.yaml.")

phase(7,"Active Speaker Matching and Root-Cause Correction",
"Link anonymous voice turns to visible face tracks only when temporal and mouth-motion evidence is sufficient.",
"The matcher checks turn/track overlap and track confidence. Mouth activity is sampled at 5 FPS, requiring three samples, and lower-face motion is compared with an upper-face control. Interpolation gaps over 1.5 s are rejected. Contrast thresholds are 0.012 minimum and 0.055 strong. Weak evidence returns unknown.",
"A prior implementation used a hard-coded video path unrelated to the current job, making frame timing invalid. The current phase resolves the source via the per-job video ID template, enabling alignment to the correct 261.5 s clip.",
"In the corrected run, 15 of 74 turns were provisionally matched (20.27%); 59 remained unknown; 126 tracks were available; zero persistent speaker-to-character mappings were accepted. Final validation still flags 64 unmatched visual turns and one missing speaker ID.",
"This is an uncalibrated heuristic, not identity accuracy. Unknown fallback avoids forced guesses but leaves attribution unresolved. The current result shows conservative handling rather than a solved speaker-identification task.",
"Artifacts: results/phase7/876c14763e77_metrics.json and data/speaker; config: configs/phase7.yaml.")

phase(8,"Keyframe Selection, Conversation Grouping, and Story Plan",
"Choose a compact chronological visual sequence, preserve dialogue, and add visual beats.",
"Candidates are ranked with dialogue semantics, speaker evidence, quality, saliency, action, and diversity. Dialogue weights in YAML are 0.35, 0.25, 0.20, 0.10, 0.10, and 0.10 respectively. Visual-only candidates use a separate quality/saliency/motion/action/diversity mix. Short dialogue windows, a six-frame group cap, and a 50 s scene span constrain selection.",
"The story plan holds timestamps and scene groups. Dialogue coverage can exceed the page budget instead of dropping utterances. Visual-only panels support silent action.",
"The run selected 79 keyframes and 79 story beats across 8 scenes. All 110 dialogue segments were covered across 70 dialogue panels; 7 visual story beats and 5 visual-only panels were recorded. One candidate without a real timestamp was dropped. Phase 8 reported zero selected duplicate frame IDs.",
"Final assembly later reported five repeated IDs, showing that uniqueness at selection does not guarantee uniqueness after continuation panels are added. Coverage also does not certify frame relevance.",
"Artifacts: data/keyframes and data/story for job 876c14763e77; results/phase8 metrics; config: configs/phase8.yaml.")

phase(9,"Multimodal Emotion Estimation",
"Attach an optional emotional cue from text, audio, and visible faces.",
"The configured models are j-hartmann/emotion-english-distilroberta-base, superb/wav2vec2-base-superb-er, and dima806/facial_emotions_image_detection. Weights are 0.45 text, 0.30 audio, 0.25 face. Missing modalities are masked and remaining weights renormalized.",
"Audio has duration and RMS filters; face sampling is limited. Labels support review and presentation but are not psychological ground truth.",
"All 110 segments received an estimate; text was available for 110, audio for 109, face for 7, and all three for 7. Mean fused confidence was 0.5562, range 0.2641–0.9635. Neutral was most frequent (62).",
"There are no reference emotion annotations. Confidence is a model output, not a validated probability. Face evidence exists for only 6.36% of segments.",
"Artifacts: data/emotions and results/phase9/876c14763e77_metrics.json; config: configs/phase9.yaml.")

phase(10,"Comic Page Layout and Face-Aware Cropping",
"Place selected panels and retain important scene and speaker regions inside the crop.",
"Layout uses target panel ratio 1.35, six panels per page, and speaker-region weighting. When multiple matched speakers occur, cropping can retain their combined face bounds. A temporal shift tolerance constrains alternate-frame selection.",
"The final layout uses 1200 × 1200 pages. Automated ROI retention averages 0.9596, minimum 0.4809; aspect error averages 0.000726; the binary face-cut counter is zero.",
"The output contains 14 pages and 80 final panels (79 selected keyframes plus a continuation panel).",
"ROI depends on upstream face boxes and geometric estimates. Minimum retention of 0.4809 indicates material loss for a detected region despite the face-cut count. Neither metric establishes aesthetic balance.",
"Artifact: data/layout/876c14763e77.json and Phase 10 log; config: configs/phase10.yaml.")

phase(11,"Panel Stylization",
"Apply a consistent comic visual treatment while preserving panel dimensions.",
"The cinematic_comic_grade_v2 profile applies bilateral smoothing (diameter 9, sigma 75), 18 color clusters, and edge emphasis (block size 9, C=9, strength 1.6). A fixed seed and cache support repeatability and faster reruns.",
"Styled images are stored separately from source frames, preserving a traceable original. Logs report cache status and edge-overlap diagnostics.",
"All 79 selected panels styled successfully with zero failures. Summary metrics report 36 cache hits and about 0.0187 s mean uncached time per panel.",
"Edge overlap and color-histogram distance are diagnostics, not reader preference or semantic preservation. Cached diagnostics can be near zero because no new filtering was performed.",
"Artifacts: data/outputs/panels_styled/876c14763e77 and phase11 logs; config: configs/phase11.yaml.")

phase(12,"Assembly, Balloons, Summaries, and Validation",
"Combine panels, original transcript text, available speaker labels, summaries, and export formats.",
"Renderer settings use 1200 × 1200 pages, six panels per page, adaptive bubble dimensions, a maximum bubble-area ratio of 0.42, and duplicate segment placement disabled. No fixed bubble-count cap is configured so dialogue can be preserved. Scene and panel summaries sit outside balloons.",
"Validation checks transcript coverage and duplication, text/frame mismatches, summaries, bounds, overlap, tails, density, and ending preservation. A tail is associated with a speaker location only where mapping evidence exists.",
"Final output reports 14 pages, 80 panels, 89 bubbles, 110 segments placed, 8 scene summaries, 79 panel summaries. No transcript loss or duplication, text mismatch, dialogue-frame mismatch, measured overlap, out-of-panel bubble, or missing/misdirected tail was recorded. Ending gap is 1.076 s.",
"Status is review_required: five frame IDs repeat, 64 turns are visually unmatched, one speaker ID is missing, and one panel has no visible listener. Geometry checks cannot prove semantic correctness.",
"Artifacts: output/876c14763e77/page.json, story_validation.json, dialogue_mapping.json, scene_summaries.json; config: configs/phase12.yaml.")

# 26
page("Chapter 6","Data Contracts, Conversation Order, and Scene Summaries",[
"Phase boundaries use timestamped records. A transcript item carries dialogue ID, text, start/end, and optional speaker ID. Diarization stores anonymous turn IDs and intervals. Face observations contain time, box, confidence, track, and optional identity cluster. Alignment links a dialogue ID with candidate frames and similarity. Story records connect panel ID, timestamp, frame ID, scene ID, and dialogue references.",
"Final page data joins geometry, source frame, dialogue, speaker state, emotion annotation, and scene/panel description. Stable IDs enable missing, duplicate, or mismatched records to be detected. Validation distinguishes coverage from identity confidence.",
"Conversation order is chronological: timestamps define sequence. Short gaps can be grouped; scene span and action/topic cues split groups. There is no evidence that the pipeline performs general discourse understanding. Grouping is temporal and heuristic.",
"Summaries should separate visual observation from inferred context. A panel sentence should describe what is visible; dialogue may support context but should not invent an action. Current outputs have full summary-field coverage, but some are generic (“visible action is unclear”). Presence checks do not establish usefulness.",
"Future schema improvement: record provenance per field (model, heuristic, or human correction) and explicit confidence/unknown states. This makes uncertainty and editorial review auditable."
],tab=(["Record","Core fields"],[
["Dialogue","ID, text, start/end, optional audio speaker"],
["Face track","ID, timestamped boxes, confidence, identity cluster?"],
["Panel","ID, time, frame ID, scene ID, dialogue IDs"],
["Summary","scene/panel ID, concise text, evidence source"],
["Bubble","dialogue IDs, speaker state, geometry, tail target"]
],[1.45,4.8]))

# 27
page("Chapter 7","Application and Implementation",[
"The frontend is standalone HTML/CSS/JavaScript, served by Flask. The server validates uploads, creates a job record and copied config, starts main.py as a child process, scans stage logs, exposes cancellation, and serves previews and downloads. README documents python app.py and http://127.0.0.1:8501.",
"Supported input types are MP4, MOV, MKV, AVI, and WEBM, up to 512 MB. Users can choose language, model size, device, download behavior, and panels per page. The UI displays stage progress and logs and provides page navigation, zoom, full-screen view, transcript/stage detail, report, and exports.",
"Pipeline remains in Python; the web layer orchestrates review. Saved per-job YAML supports reproducibility. Required infrastructure includes Python, FFmpeg/FFprobe, model dependencies, and access to gated pyannote models.",
"First model load can be slow; Phase 5 is especially expensive on CPU. The current app allows one active job to control memory use. Cancellation preserves completed artifacts but an interrupted write can be incomplete; a fresh job ID is the recovery mechanism.",
"Core files: app.py; src/web_server.py; src/web_jobs.py; main.py; web/index.html, app.js, styles.css."
])

# 28
page("Chapter 8","Experimental Setup and Evaluation Protocol",[
"The completed job 876c14763e77 uses a 261.545 s H.264 clip at 1280 × 544, 23.97602 FPS, 6271 frames, and 16 kHz audio of 261.503 s. The saved source hash is e87653a2f73ddf46725f6fd78989e2ab6061466215b7f738b8661e5a2c4e710b.",
"The run used faster-whisper small/English/CPU int8; pyannote diarization 3.1; InsightFace buffalo_l on CPU; OpenCLIP ViT-B-32; and three emotion models. Stage runtimes are descriptive and depend on hardware, model startup, and cache.",
"Evaluation has a structural layer and a content layer. Structural tests check artifact consistency, timestamps, segment coverage, page data, and geometric rules. Content evaluation would compare outputs with annotated transcript, speaker turns, identities, frame relevance, emotion, and summaries. No such reference labels were available.",
"Configured gates are operational criteria, not scientific quality thresholds. Therefore this report records counts and consistency checks while withholding unsupported accuracy claims.",
"Reproduction identifiers: video ID 876c14763e77; source SHA-256 above; saved configs under data/jobs/876c14763e77/configs."
],tab=(["Item","Recorded setup"],[
["Video","261.545 s; 1280 × 544; 24000/1001 FPS"],
["ASR","faster-whisper small; English; CPU int8; VAD"],
["Diarization","pyannote/speaker-diarization-3.1"],
["Face","InsightFace buffalo_l; CPU; about 1 FPS"],
["Alignment","OpenCLIP ViT-B-32, laion2b_s34b_b79k"],
["Final output","14 pages; 80 panels; PDF/CBZ/HTML/PNG"]
],[1.65,4.55]))

# 29
page("Chapter 9","Quantitative Results",[
"Table 9.1 summarizes values directly stored in the current run artifacts. Coverage means a record was carried forward; it does not mean that the record was correct."
],tab=(["Stage / check","Observation","Meaning"],[
["Ingestion","6271 frames; A/V gap 0.042 s; 20/20 reads","Integrity gates passed"],
["ASR","110 segments; no drops; RTF 0.5094","No reference WER/MAE"],
["Diarization","74 turns; 3 speakers; 110 labels","No reference DER"],
["Frame pool","49 shots; 355 / 637 kept","No boundary labels"],
["Face analysis","266 detections; 126 tracks","Counts only"],
["Phase 7","15/74 matched; 59 unknown; 0 persistent maps","Low visible speaker coverage"],
["Phase 8","79 keyframes; 110/110 dialogue covered","Coverage, relevance unscored"],
["Emotion","Face available for 7/110; mean score 0.5562","No emotion labels"],
["Layout/styling","14 pages; ROI mean 0.9596; 79 styled","Diagnostic metrics"],
["Assembly","80 panels; 89 bubbles; review_required","Structural checks plus issues"]
],[1.25,2.5,1.95]),note="Source: per-stage metrics and output/page.json for job 876c14763e77.")

# 30
page("Chapter 9","Qualitative Review of the Exported Comic",[
"Figure 9.1 shows the first page of the generated comic. It has six panels and readable balloon text, but visual inspection identifies important weaknesses: the opening panel shows a coastal scene before the workshop sequence; several captions say that the action is unclear; and later monitor images do not obviously depict the adjacent dialogue.",
"Visible speakers are labeled generically (“Speaker 1”), not as character identities. Some balloons are readable, yet readers cannot consistently infer which character speaks. This qualitative evidence is consistent with the 20.27% provisional Phase 7 match rate and the final unmatched-speaker warning.",
"This is an illustrative page inspection, not a formal reader study. A stronger evaluation should use multiple reviewers and score panel relevance, summary faithfulness, speaker clarity, reading order, and story comprehension."
],img=ROOT/"output"/"876c14763e77"/"page_01.png",note="Figure 9.1. Page 1 from output/876c14763e77/page_01.png.")

# 31
page("Chapter 9","Interpreting Coverage and Attribution",[
"Complete transcript placement contrasts with limited visual speaker matching. The renderer can preserve recognized speech while leaving uncertain speaker identity unresolved. This is preferable to dropping dialogue or guessing based solely on visibility, but it leaves the reader’s central attribution question unanswered in many panels.",
"Scene and panel summary fields are present for every record counted in validation. This verifies schema coverage. It does not verify whether summaries are distinct, concise, faithful, or informative. Likewise, zero measured bubble overlaps indicates the layout checker found no geometric collision; it does not establish a natural reading order.",
"A percentage chart can normalize record counts but cannot make unlike measures comparable. Transcript placement is a coverage rate, speaker matching is evidence availability, and emotion modality coverage is data presence. They are not model accuracy."
],tab=(["Measure","Numerator / denominator","What it says"],[
["Transcript placement","110 / 110","All recognized segments were placed"],
["Scene summaries","8 / 8","Fields present"],
["Panel summaries","79 / 79","Fields present"],
["Speaker match","15 / 74","Turns with provisional visible-track match"],
["All emotion modalities","7 / 110","Segments with face+audio+text evidence"]
],[1.55,1.85,2.6]))

# 32
page("Chapter 10","Verification and Test Results",[
"The recorded unit/regression run completed with 47 passed and 18 skipped in 20.24 seconds. Covered modules include ingestion, diarization utilities, tracking, Phase 7 matching, Phase 8 story planning, Phase 9 fusion, Phase 12 rendering/validation, and web job behavior. Skipped tests are environment-dependent, not successful executions.",
"Current output checks found 110/110 transcript segments placed, no missing or duplicated segments, no text/frame mismatch, 8 scene summaries and 79 panel summaries, no measured balloon overlap or overflow, and no missing or misdirected tails. The ending frame occurs 1.076 seconds after last dialogue.",
"Review findings remain: frame IDs s24_f017, s30_f019, s31_f024, s31_f065, and s34_f005 repeat; 64 speaker turns are visually unmatched; one speaker ID is missing; and one panel lacks a visible listener. Overall status is review_required.",
"Tests cover software paths and output invariants; they do not guarantee semantic correctness across videos. Manual review of the comic is still necessary."
],tab=(["Validation","Outcome","Status"],[
["Pipeline","All 12 stages completed","Pass"],
["Dialogue","110 placed; none missing/duplicated","Pass"],
["Text/frame","Zero mismatches","Pass"],
["Summaries","8 scenes; 79 panels; none missing","Pass"],
["Bubble geometry","0 overlap/overflow/tail errors","Pass"],
["Frame repetition","5 repeated IDs","Review"],
["Speaker relation","64 unmatched; 1 ID missing; listener absent in 1","Review"],
["Overall","review_required","Review"]
],[1.55,3.25,1.0]))

# 33
page("Chapter 10","Discussion, Limitations, and Threats to Validity",[
"The strongest result is end-to-end integration: one clip traversed all phases and produced pages, dialogue, summaries, exports, and explicit validation findings. Ingestion checks passed; all detected transcript segments reached the comic; and geometric checks were clean.",
"The principal weakness is audio-to-visual identity matching. Only 20.27% of turns matched a visible track and no persistent mapping was accepted. The unknown fallback reduces unsupported assignments but leaves attribution unresolved. This is the main limit on conversational clarity.",
"Frame and summary quality need human review. Five repeated frame IDs appear after final assembly even though Phase 8 reported no selected duplicate IDs. Generic descriptions satisfy non-empty checks without being useful narrative summaries.",
"Validity is limited by one clip from one scene type; no annotated transcript, speaker, face, shot, action, or emotion labels; model outputs feeding later models; CPU and cache effects; and possible selection bias. The run cannot establish generalization, comprehension, or improvement over a human baseline.",
"Next steps: build an annotated evaluation set; validate speaker matching on held-out videos; deduplicate after continuation panels; measure WER, DER, identity precision/recall, frame relevance, and summary faithfulness; conduct blinded reader studies; and retain human-correctable identity mapping."
])

# 34
page("Chapter 11","Conclusion and Future Work",[
"CineComic-MML implements a modular workflow for video-to-comic generation. Twelve stages handle media integrity, speech, diarization, shot/frame selection, face tracks, multimodal alignment, active-speaker heuristics, story keyframes, emotion, layout, styling, and final rendering. A Flask app provides local upload and review.",
"The recorded run demonstrates operational feasibility: a 261.545-second clip generated 14 pages and 80 panels, with 110 transcript segments placed, eight scene summaries, and 79 panel summaries. Media-integrity checks and several output-geometry checks passed.",
"The project does not yet reliably answer “who is speaking?” The 15/74 provisional match rate, zero accepted persistent mappings, repeated frame warnings, and absence of ground-truth evaluation are material limitations. The evidence supports an integrated system, but semantic quality remains unproven and visible defects remain.",
"Priority work is to annotate diverse clips, replace or supplement mouth-motion heuristics with a validated active-speaker model, support manual character correction, enforce post-assembly frame uniqueness, track summary provenance, quantify confidence calibration, and evaluate reader comprehension."
])

# 35
page("References","Research Sources",[
"[1] A. Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision,” arXiv:2212.04356, 2022. https://arxiv.org/abs/2212.04356",
"[2] H. Bredin, “pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe,” Interspeech 2023. doi:10.21437/Interspeech.2023-105. https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.pdf",
"[3] A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML 2021, PMLR 139, pp. 8748–8763. https://arxiv.org/abs/2103.00020",
"[4] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “ArcFace: Additive Angular Margin Loss for Deep Face Recognition,” CVPR 2019, pp. 4690–4699. https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html",
"[5] J. L. Alcazar et al., “Active Speakers in Context,” CVPR 2020, pp. 12465–12474. https://openaccess.thecvf.com/content_CVPR_2020/html/Alcazar_Active_Speakers_in_Context_CVPR_2020_paper.html",
"[6] S.-W. Yang et al., “SUPERB: Speech processing Universal PERformance Benchmark,” Interspeech 2021. https://arxiv.org/abs/2105.01051",
"[7] X. Yang et al., “Automatic Comic Generation with Stylistic Multi-page Layouts and Emotion-driven Text Balloon Generation,” ACM TOMM, 17(2), Article 55, 2021. doi:10.1145/3440053. https://doi.org/10.1145/3440053",
"These sources provide research context; the project does not claim to reproduce the complete methods or reported results of the cited papers."
])

# 36
page("References","Project Artifacts Consulted",[
"Implementation and method sources: CineComic_MML/README.md; main.py; configs/phase1.yaml through phase12.yaml; src/ modules for ingestion, audio, vision, speaker matching, keyframes, emotion, layout, comic rendering, and the web server.",
"Recorded run sources: data/jobs/876c14763e77/job.json and copied configs; data/metadata, subtitles, diarization, shots, faces, tracks, alignment, speaker, keyframes, story, emotions, and layout; results/phase5, phase7, phase8, phase9; data/*metrics.json.",
"Final-output sources: output/876c14763e77/page.json; story_validation.json; dialogue_mapping.json; scene_summaries.json; page images; comic.pdf and comic.cbz.",
"The report describes the artifacts associated with job 876c14763e77. Different model versions, source video, hardware, cache state, or configuration can change the output. An older PROJECT_REPORT.md refers to a different job and was not used as the source for current-run metrics."
])

# 37
page("Appendix A","Reproduction and Local Run Guide",[
"Prerequisites: install Python 3.10/3.11+, FFmpeg/FFprobe, and project requirements. Create and activate a virtual environment; install requirements.txt; run setup_project.py if required folders are absent; then run env_check.py and review results/environment_report.json. Configure gated pyannote model access without storing tokens in the report.",
"CLI examples from the repository:",
"python main.py --config configs/phase1.yaml --video path/to/clip.mp4 --video-id clip_001 --stages ingestion",
"python main.py --config configs/phase2.yaml --video path/to/clip.mp4 --video-id clip_001 --stages all",
"To launch the local web app, run python app.py from CineComic_MML and open http://127.0.0.1:8501. Upload a supported video, follow phase progress, then inspect pages, transcript, and validation before exporting.",
"For reproducible comparison, record source SHA-256, job ID, all copied YAML, Python/model versions, hardware, elapsed times, and cache state. Raw runtime is not comparable unless those conditions are recorded.",
"Artifacts are stored under data/uploads, data/jobs, data, results, and output. Respect repository policy for large media and generated files."
])

# 38
page("Appendix B","Detailed Result Ledger and Evaluation Plan",[
"Compact ledger: ingestion passed with zero frame-count difference, 0.042 s A/V gap, no decode errors, and 20/20 seeks; ASR produced 110 segments but no reference WER; diarization produced 74 turns over three anonymous IDs but no reference DER; extraction retained 355 candidates across 49 shots; face analysis made 266 detections and 126 tracks; Phase 7 matched 15/74 turns and retained 59 unknown; Phase 8 selected 79 keyframes with full dialogue coverage; emotion had face evidence for seven of 110 segments; final assembly produced 14 pages, 80 panels, and 89 bubbles.",
"The recorded test run reported 47 passed and 18 skipped. Test modules include ingestion, diarization, tracking, Phase 7/8/9/12, story planning, and web jobs. This does not imply coverage of every real-world scenario.",
"Recommended benchmark extension: annotate separate training/development/test videos; calculate transcript WER; diarization DER with declared collar; speaker-face precision/recall; frame relevance at k; timestamp alignment error; duplicate-panel rate; summary faithfulness ratings; and blinded reader attribution/comprehension."
],tab=(["Measure","Current evidence","Needed next"],[
["Transcript","Not accuracy-scored","Reference transcript; WER"],
["Diarization","Not accuracy-scored","Reference turns; DER"],
["Speaker identity","15/74 provisional matches","Verified audio-face labels"],
["Frame relevance","Not scored","Human panel relevance labels"],
["Summary quality","Field coverage only","Faithfulness/usefulness ratings"],
["Reader clarity","No study","Blinded comprehension study"]
],[1.45,2.1,2.4]))

doc.core_properties.title="Multimodal Video to Comic Generation System — Final Year Project Report"
doc.core_properties.subject="Phase-wise methodology and results for CineComic-MML"
doc.core_properties.author="[Student Name]"
doc.core_properties.keywords="CineComic-MML, multimodal, video-to-comic, final year project"
OUT.parent.mkdir(parents=True,exist_ok=True)
doc.save(OUT)
print(f"Wrote {OUT}")
