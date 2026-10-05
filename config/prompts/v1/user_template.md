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
Using the frames and the labels above, write exactly {n_questions} tough, agentic questions following every rule
in your instructions: three distinct categories, at least one temporally grounded and at least one requiring
clinical judgment, varied answer types, at most one multiple-choice question with 4-5 options `A. ...`.
Each question must be answerable from the full video by an agent with seek/zoom/count/measure/look-up tools,
and verifiable by a clinician from the video plus these labels. Cite evidence as time ranges in `MM:SS` form
in the text and as seconds in `evidence_timestamps`. Do not reference frame numbers.

Return only the JSON object required by the schema.
