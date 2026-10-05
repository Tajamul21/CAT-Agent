# Surgical video for question generation

## Dataset
**{dataset_title}** - {dataset_blurb}
Licence: {license}
Citation: {citation}
{dataset_notes}

## Video metadata
{metadata_block}

## Expert label timeline
{timeline_section}

{glossary_section}

{timeline_note_section}

## Frames
{n_frames} sampled frames follow in chronological order. Each frame is preceded by its caption
(`Frame k/n - t=MM:SS.s - <label at t>`) and carries the same time code burnt in at the bottom left.
Times are in the displayed video timeline described above. Frames are sparse: events can occur between them,
so cite time as ranges and rely on the label timeline for exact boundaries.
<<FRAMES>>
## Task
Write exactly {n_questions} situation-awareness questions about this case, following every rule in your
instructions:
- q1 L1 Perception, q2 L2 Comprehension, q3 L3 Projection (forecast first, then verify);
- each about a different phase or transition, with three different categories and clinical uses where possible;
- at least one one-word or very short closed answer and at least one multi-line answer;
- at most one question that asks for a time, an interval or a duration; anchor the rest on surgical events;
- closed answer sets in the question and in `options` wherever possible, with "not determinable" allowed
  when the view may not settle it.
Each question must be answerable only by watching this video, by an agent with seek, zoom, count, measure and
look-up tools that does not see these labels, and verifiable by a clinician. Record evidence as time ranges in
`evidence_timestamps`. Never mention frames or frame numbers.

Return only the JSON object required by the schema.
