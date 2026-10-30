"""Step 2 - visual extraction: the extractor model reads the page and returns PlanFeatures."""

from __future__ import annotations

from app.core.config import get_settings
from app.core.schemas import PlanFeatures, Usage
from app.pipeline.llm import image_block, invoke_structured
from app.pipeline.render import PageImage

# Short, fixed label definitions so preliminary_label uses our classes.
# The detailed rules and confusion cases come from retrieval, for the decision step.
LABEL_GUIDE = """\
- architectural: floor plans, elevations, sections whose subject is the building itself (walls, doors/windows, room names, finishes); prefix A-
- structural: grid lines, columns, beams, foundations, framing, rebar details, load notes; prefix S- or F- (foundation)
- electrical: outlets, switches, light fixtures, circuits, panels, single-line diagrams; prefix E-
- plumbing: water supply and drainage pipes, sanitary fixtures, risers, slopes; prefix P-
- hvac_mechanical: ducts, diffusers/grilles, AHU, fan coils, airflow values, HVAC piping; prefix M-
- civil_site: site layout, contour lines, grading, roads, parking, site utilities, parcel limits; prefix C-
- fire_safety: sprinklers, fire alarm devices, evacuation routes and signs, extinguishers, fire compartments; prefix FP-, SP-, FA-
- not_a_plan: photos, invoices, letters, forms, map/app screenshots, sketches without scale or title block"""

SYSTEM_PROMPT = f"""You are an expert in reading construction drawings.
You receive a document page as two images: the full page, then a zoomed crop of its
bottom-right corner where a title block usually is (the crop may contain no title block).

Extract only what is actually visible:
- Copy title-block text, sheet number and scale exactly as printed. Use null / empty when
  something is absent or unreadable; never invent or complete text.
- sheet_title: the drawing title written in the title block (e.g. "ELECTRIC POWER PLAN &
  LEGEND", "FIRST FLOOR PLUMBING PLAN"), with any discipline word printed with it (e.g.
  "STRUCTURAL THIRD FLOOR PLAN", "SPRINKLER AUDITORIUM PLAN"). It is never a room or area
  label from inside the drawing (e.g. "MECHANICAL EQUIPMENT ROOM") and never the project,
  building, client, firm or address. If the title block has no drawing title, use the
  title printed under the main drawing.
- sheet_number: if the page shows both an internal drawing number (e.g. "03-5908") and a
  discipline sheet number (e.g. "E-2", "M-1", "FP-2"), use the discipline sheet number.
- legend_items: entries of the legend or symbol key only. Title-block content (firm,
  address, phone, approvals), revision notes, general notes and stray numbers are not
  legend items. At most 15.
- detected_symbols: the distinct technical elements you can actually see drawn on this sheet
  (equipment, devices, pipe / duct / wiring runs, structural members, site features...),
  named as on the drawing. List nothing you cannot see. Drawing furniture such as
  dimensions, notes, scale, title block or sheet number are not symbols. At most 15.
- description: 2-4 factual sentences naming the drawing type and the systems drawn
  (pipes, ducts, sprinklers, circuits, footings...). Do not use class names such as
  "architectural plan"; describe what is shown. Leave out project, building, client and
  firm names (they are already in title_block_text).

Then give a preliminary_label from these classes:
{LABEL_GUIDE}

How to choose the class:
- Every discipline is drawn over the building's floor outline, so walls and rooms alone do
  not make a page architectural.
- Decide from, in this order: the sheet title (e.g. "PLUMBING PLAN", "SPRINKLER PLAN",
  "FOUNDATION PLAN"), the sheet number prefix, then the system that is the subject of the drawing.
- Use architectural only when the building itself is the subject and no other system is.
- If the page is not a technical drawing at all, use not_a_plan."""


def page_images(page: PageImage) -> list[dict]:
    """The two image blocks every model call gets: full page + title-block crop."""
    return [
        {"type": "text", "text": "Full page:"},
        image_block(page.image, page.media_type),
        {"type": "text", "text": "Zoomed bottom-right corner (title block area):"},
        image_block(page.title_block, page.title_block_media_type),
    ]


def extract_features(page: PageImage) -> tuple[PlanFeatures, Usage]:
    """Return the page features and the token usage of the call(s)."""
    return invoke_structured(
        get_settings().sonnet_model_id,
        PlanFeatures,
        SYSTEM_PROMPT,
        page_images(page),
        "Extract the page features.",
        max_tokens=2048,
    )
