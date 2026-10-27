# PlanSense discipline rules

Reference knowledge used for retrieval. Each `## <label>` section below is indexed as one
rule document (metadata: `type=rule`, `id=<label>`). Keep the heading text equal to the label.
A section that covers very different kinds of drawings can be split with `### <part>`
headings: each part is indexed as its own rule (with the section intro), same `id`.

General principles that apply to every page:
- Decide first whether the page is a technical drawing at all (title block, scale, drawing
  conventions such as linework, dimensions, grid lines, symbols, legend). If not -> not_a_plan.
- Sheet number prefix is strong evidence, but the content of the sheet wins when they clearly
  disagree (e.g. a life-safety plan numbered A-005 is fire_safety).
- Most drawings of one building share the same floor outline. The discipline is told apart by
  what is drawn ON TOP of the outline (fixtures, ducts, pipes, circuits, rebar...) and by the
  legend, not by the walls themselves.
- French project conventions also appear: lots (ARC, STR/GO, ELEC/CFO/CFA, PLB, CVC, VRD, SSI),
  levels (SS = sous-sol, RDC = rez-de-chaussée, R+1...), "plan de masse" (site plan),
  "coupe" (section), "façade" (elevation), "cartouche" (title block).

## architectural

Architectural drawings define the building itself: spaces, walls, openings and finishes.

Sheet prefix and titles: A- (A-101, A1.01), also I- (interiors). Titles such as "Floor Plan",
"Ground Floor Plan", "Level 2 Plan", "Reflected Ceiling Plan", "Elevations", "Building Sections",
"Wall Sections", "Door Schedule", "Window Schedule", "Roof Plan", "Enlarged Plan", "Finish Plan".
French: "Plan RDC", "Plan R+1", "Façades", "Coupes", "Plan de toiture", lot ARC / Architecte.

Visual clues:
- Walls drawn as double lines or hatched/poché, with door swings (quarter-circle arcs) and windows.
- Room names with room numbers and areas (m² / sq ft), sometimes finish tags.
- Dimension strings along the outside of the building, section and elevation markers (circles
  with arrows), stairs with up/down arrows, furniture and sanitary fixture outlines for layout only.
- Elevations show facades with materials, levels (+0.00, FFL), window patterns.
- Door/window schedules: tables of door numbers, sizes, types, fire ratings.

Common confusions:
- vs structural: architectural shows rooms, doors, names and finishes; structural shows grid,
  columns, beams, slab thickness and rebar with few or no room names.
- vs plumbing/electrical/hvac: those reuse the architectural background as light grey and draw
  their own systems in dark lines. If the dominant content is pipes, circuits or ducts, it is not
  architectural.
- vs fire_safety: a floor plan whose main purpose is egress routes, fire ratings and compartments
  is fire_safety.
- A sketch or 3D rendering without scale or title block is not_a_plan.

## structural

Structural drawings define the load-bearing system: foundations, frames, slabs and their
reinforcement.

Sheet prefix and titles: S- (S-101, S2.01). Titles such as "Foundation Plan", "Framing Plan",
"Level 2 Framing", "Roof Framing Plan", "Column Schedule", "Beam Schedule", "Typical Details",
"Rebar Details", "General Structural Notes". French: lot GO (gros oeuvre) / STR, "Plan de
coffrage", "Plan de ferraillage", "Plan des fondations", bureau d'études structure.

Visual clues:
- Structural grid lines with bubbles (A, B, C... / 1, 2, 3...) dominate the sheet.
- Columns as filled squares/circles or steel I-sections at grid intersections; beams as lines
  with size tags (W12x26, IPE 300, HEA, 30x60).
- Footings, pile caps, foundation walls, slab edges; slab thickness and level annotations.
- Reinforcement details: rebar bars with diameters and spacing (HA12 @ 200, #5 @ 12" o.c.),
  stirrups, sections through beams and footings, anchor bolts, base plates, connections.
- Load notes: live/dead loads, concrete grade (C25/30, f'c), steel grade, soil bearing pressure.

Common confusions:
- vs architectural: structural has few room names, no furniture, no door swings; it is about
  grid, members and reinforcement.
- vs civil_site: civil shows terrain, roads and networks outside the building; structural
  foundations stay within the building footprint.

## electrical

Electrical drawings show power distribution, lighting, and low-current systems.

Sheet prefix and titles: E- (E-101, E2.01), also T- (telecom) and sometimes FA- for fire alarm
inside the electrical lot. Titles such as "Lighting Plan", "Lighting Layout", "Power Plan",
"Receptacle Plan", "Single-Line Diagram", "Riser Diagram", "Panel Schedules", "Grounding Plan",
"Low Voltage / Data Plan". French: lot ELEC, CFO (courants forts), CFA (courants faibles),
"Plan d'éclairage", "Plan prises de courant", "Schéma unifilaire", "Tableau électrique (TGBT)".

Visual clues:
- Receptacles/outlets (circles with two lines), switches (S, S3), light fixtures (squares with
  an X, circles, rectangles for troffers/luminaires), exit signs, emergency lighting.
- Circuit references and homeruns: arrows with tags like C1, L1-3, LP-1-12; curved wiring lines
  between fixtures and switches.
- Panels, distribution boards, transformers, switchboards; single-line diagrams with breakers,
  feeders, ratings (A, kVA, V).
- Panel schedules: tables of circuits, breaker sizes, loads.
- Legend with electrical symbols (sockets, switches, luminaires, data outlets, cable trays).

Sub-disciplines: lighting, power, low-current (data, telephone, security, access control, CCTV).

Common confusions:
- vs hvac_mechanical: HVAC shows ducts (thick double lines with airflow), electrical shows thin
  circuit lines and small symbols.
- vs fire_safety: a dedicated fire alarm plan (detectors, manual call points, sounders) or an
  evacuation plan is fire_safety; smoke detectors appearing on an otherwise power/lighting plan
  stay electrical.

## plumbing

Plumbing drawings show water supply, drainage and sanitary systems.

Sheet prefix and titles: P- (P-101, P2.01). Titles such as "Plumbing Plan", "Domestic Water
Plan", "Sanitary / Waste and Vent Plan", "Storm Drainage", "Plumbing Riser Diagram",
"Isometric", "Fixture Schedule", "Water Heater Detail". French: lot PLB (plomberie), "Plan de
plomberie", "Évacuations EU/EV/EP", "Alimentation EF/ECS" (eau froide / eau chaude sanitaire),
"Colonnes montantes".

Visual clues:
- Pipes as single lines with line-type codes: CW (cold water), HW (hot water), HWR, W/WS (waste,
  soil), V (vent), SD/ST (storm), G (gas); French EF, EC/ECS, EU, EV, EP.
- Sanitary fixtures highlighted and tagged (WC, LAV, SINK, UR, SH, FD floor drain).
- Riser and isometric diagrams: vertical pipe stacks with pipe diameters (DN50, 2"), valves,
  cleanouts (CO), traps.
- Slope annotations for drainage (1%, 1/4"/ft), invert levels, floor drains, water heaters,
  backflow preventers, meters.

Common confusions:
- vs hvac_mechanical: HVAC pipes (chilled/heating water CHW/HHW) and ducts belong to HVAC;
  domestic water, drainage and sanitary fixtures belong to plumbing.
- vs fire_safety: sprinkler piping and sprinkler heads are fire_safety even though they are pipes.
  Fire water (FW) lines, standpipes and hose racks are fire_safety; domestic water (CW/HW)
  and drainage to fixtures are plumbing.
- vs civil_site: networks outside the building (site sewer, water mains, manholes in roads)
  are civil_site.

## hvac_mechanical

HVAC / mechanical drawings show heating, ventilation, air conditioning and related equipment.

Sheet prefix and titles: M- (M-101, M2.01), also H-. Titles such as "HVAC Plan", "Mechanical
Plan", "Ductwork Plan", "HVAC Piping Plan", "Mechanical Roof Plan", "Equipment Schedule",
"Air Flow Diagram", "Controls". French: lot CVC (chauffage, ventilation, climatisation),
"Plan de ventilation", "Réseaux aérauliques", "CTA" (centrale de traitement d'air), "VMC".

Visual clues:
- Ductwork as thick double lines with sizes (600x300, 24x12, Ø250), elbows, transitions, fire
  dampers, volume dampers.
- Air terminals: supply diffusers (squares with an X or arrows), return/exhaust grilles, with
  airflow values (m³/h, CFM, l/s).
- Equipment: AHU / CTA, fan coil units (FCU, VCV), VAV boxes, rooftop units (RTU), fans,
  chillers, boilers, heat pumps, split units; equipment schedules with capacities (kW, BTU).
- Hydronic piping for heating/cooling (CHW, CHWR, HHW, HWS/HWR), refrigerant lines.

Common confusions:
- vs plumbing: ducts and air terminals are always hvac; domestic water and sanitary drainage are
  plumbing.
- vs electrical: thin wiring and fixture symbols are electrical; thick ducts with airflow are hvac.

## civil_site

Civil / site drawings show the land around the building: terrain, roads, parking, utilities and
parcel boundaries.

Sheet prefix and titles: C- (C-101, C1.0), also V- (survey) and L- (landscape). Titles such as
"Site Plan", "Grading Plan", "Topographic Survey", "Utility Plan", "Storm Drainage Plan",
"Erosion Control", "Paving Plan", "Road Profile", "Site Details". French: lot VRD (voirie et
réseaux divers), "Plan de masse", "Plan topographique", "Plan des réseaux", "Plan de
nivellement", "Plan cadastral".

Visual clues:
- Topography: contour lines with elevation labels, spot elevations, north arrow, coordinates.
- Parcel/property limits, setbacks, easements, the building shown only as a footprint outline.
- Roads, curbs, parking stalls, sidewalks, road centerlines with stations (0+00), plan/profile
  views.
- Underground networks: storm/sanitary sewers with manholes (MH), catch basins, inverts, water
  mains, hydrants, gas and electrical duct banks in the site.
- Erosion control (silt fences), landscape planting.

Common confusions:
- vs not_a_plan: a Google Maps / web map screenshot or satellite image also shows roads and
  parcels, but it has map UI (search bar, zoom buttons, pins, place labels, imagery) and no
  title block or engineering scale -> not_a_plan.
- vs plumbing: networks inside the building are plumbing; networks in the site are civil.
- vs architectural: an architectural site plan focused on the building massing with landscape
  context may look similar; contours, grading, utilities and road geometry point to civil_site.

## fire_safety

Fire safety drawings show fire protection, fire detection and life safety / evacuation.
They come in three quite different kinds, described below; each kind is enough on its own.

### Sprinklers and standpipes

Sprinkler / fire protection piping drawings (wet or dry sprinkler systems, standpipes).

Sheet prefix and titles: FP-, SP-, F- (fire protection, sprinklers). Titles such as
"Sprinkler Plan", "Floor Plan Sprinkler", "Fire Protection Plan", "Sprinkler Riser Diagram",
"Standpipe Riser", "Fire Service". Often drawn by a sprinkler contractor ("... Automatic
Sprinkler Co.") with its own title block. French: "Plan sprinkler", "RIA" (robinet d'incendie
armé), "colonne sèche / humide".

Visual clues:
- Sprinkler heads (upright, pendent, sidewall) as small circles or dots on a regular grid,
  fed by branch lines, cross mains and feed mains; sprinkler riser with riser control valve.
- Fire water (FW) lines and fire service entry, standpipes, fire hose racks / cabinets.
- Valves and devices: OS&Y (outside screw and yoke) gate valve, PIV (post indicator valve),
  tamper switch (TS), water flow switch, alarm check valve, inspector's test and drain (DR),
  Siamese / fire department connection (FDC).
- Pipe sizes on every run, hangers and cutting dimensions, hydraulic design area or density.

Common confusions:
- vs plumbing: both are pipes and valves, but sprinkler systems carry FIRE WATER (FW) to
  sprinkler heads, standpipes and hose racks; plumbing carries domestic cold/hot water (CW/HW)
  and drainage to fixtures (WC, LAV, sinks, floor drains). Pipes feeding sprinkler heads are
  fire_safety even when they look like plumbing.
- vs hvac_mechanical: no ducts, diffusers or air handling units; piping is fire water only.

### Evacuation and escape plans

Posted evacuation / escape and rescue plans.

Titles such as "Evacuation Plan", "In Case of Emergency", "Escape Route Plan", "Fire Escape
Plan"; German "Flucht- und Rettungsplan"; French "Plan d'évacuation", "Plan d'intervention".

Visual clues:
- Simplified floor plan with green escape route arrows, green running-man exit signs,
  "You are here" / "Vous êtes ici" / "Standort" marker, assembly point.
- Red fire extinguisher, fire blanket, hydrant, fire alarm call point and first aid symbols
  (ISO 7010 pictograms), an instructions panel.
- Often no classic engineering title block or scale, but still a plan (fire_safety),
  recognisable by the safety pictograms and escape routes.

Common confusions:
- vs architectural: same floor outline, but the content is escape routes and fire equipment.
- vs not_a_plan: an evacuation plan is a plan even without a title block.

### Fire alarm and life safety

Fire detection / alarm layouts and life safety or code compliance plans.

Sheet prefix and titles: FA- (fire alarm), LS- or G- (life safety / code). Titles such as
"Fire Alarm Plan", "Life Safety Plan", "Code Compliance Plan", "Egress Plan",
"Fire Compartmentation". French: lot SSI (système de sécurité incendie), "Désenfumage",
"Compartimentage".

Visual clues:
- Fire alarm: smoke / heat detectors, manual call points / pull stations, horns / strobes,
  fire alarm control panel (FACP), loops / zones.
- Life safety: fire-rated walls drawn with distinct hatches or colors (1-HR, 2-HR, EI60),
  fire compartments, travel distances, exit widths, occupant loads.

Common confusions:
- vs electrical: a dedicated fire alarm layout is fire_safety; detectors shown among outlets
  and lighting on a power plan stay electrical.
- vs architectural: a life-safety plan numbered A-xxx is still fire_safety.

## not_a_plan

Anything that is not a construction drawing must be rejected with label not_a_plan and
is_plan = false.

Typical documents:
- Photographs: construction site photos, building photos, people, machinery, phone pictures.
- Business documents: invoices, quotes, purchase orders, receipts, letters, emails, contracts,
  forms, reports, spreadsheets, meeting minutes.
- Maps and map screenshots: Google Maps / OpenStreetMap / satellite views with map UI (search
  bar, zoom controls, pins, street and business labels, Street View images).
- Screenshots of software, websites, dashboards, chat messages.
- Hand sketches or diagrams without scale, title block, or drawing conventions; marketing
  renderings and 3D visualisations.

Visual clues:
- Continuous-tone photographic texture instead of clean linework.
- Paragraphs of prose, tables of prices and totals (VAT/TVA, amount due), letterheads,
  signatures, logos.
- No title block, no sheet number, no scale, no north arrow, no legend of drawing symbols.

Common confusions:
- A photo of a printed drawing or a scanned drawing is still a plan if the drawing content
  (title block, linework) is legible -> classify its discipline.
- An evacuation plan is a plan (fire_safety) even without an engineering title block.
- A survey map with contour lines and a title block is civil_site, not a map screenshot.
