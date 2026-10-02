# House: Centex Mesilla, Elevation R

A detailed 3D model of a two-story Centex "Mesilla" (plan 1728.100.03, 2,615 sq ft,
4 bed, 3 bath) built right-handed with the Elevation R front. It is generated from
the builder's vector floor plan and calibrated to the lot survey. It comes apart by
floor so you can look inside.

## Files

| File | What it is |
|---|---|
| `house_viewer.html` | Open in a browser. The model is embedded; three.js loads from a CDN. Has views, layer toggles, a pull-apart slider, a section-cut height slider, room labels with areas, the builder's plan linework on the floors, and click-to-identify. |
| `house.glb` | The same model for Blender, macOS Quick Look, or any glTF viewer. Meters, Y up. Each node is named `<layer>\|<part>`, for example `Floor 2\|Room: Owner's Suite`. |
| `print/1_first_floor.stl` | Slab, first floor walls, cladding, stairs, kitchen. 1:80, mm. |
| `print/2_second_floor.stl` | Floor framing and second floor walls. Sits on the first floor walls. |
| `print/3_roof_main.stl` | Main hip, front hip, and the tower gable. Sits on the second floor walls. |
| `print/4_roof_garage.stl` | Garage gable roof. |
| `plan.json` | Walls and plan linework extracted from the PDF, in feet. |
| `extract_plan.py` | PDF to `plan.json`. |
| `openings.py` | Finds doors and windows as gaps between wall ends. |
| `build_house.py` | `plan.json` to everything else. |
| `viewer_template.html` | Viewer source. `build_house.py` embeds the model into it. |

At 1:80 the footprint is 130 x 208 mm and fits a 220 mm bed. Change `PRINT_SCALE`
in `build_house.py` for another scale.

## Rebuild

```bash
uv run extract_plan.py "<path>/Centex Mesilla Elevation R Rendering and Option Floor Plan.pdf"
uv run build_house.py
```

Dependencies are inline (PEP 723) and live only in uv's cache, about 100 MB:
pymupdf, shapely, numpy, scipy, trimesh, mapbox-earcut, manifold3d. No CAD
application is needed.

## Where the numbers come from

- **Walls, doors, windows, fixtures:** page 2 of the builder's option floor plan
  PDF, which is vector line art. Wall polygons are read directly from it. The
  plan is drawn left-handed; this house is right-handed (garage on the right from
  the street, per the configuration sheet and the survey), so X is mirrored.
- **Scale and footprint:** the lot survey: 34.0 ft wide, 54.5 ft deep on the
  garage side, 44.5 ft on the study side, and the 4 ft entry recess. Computed
  room areas land within a few sq ft of the brochure's room sizes.
- **Heights:** 9 ft first floor and 8 ft second floor ceilings (Included Features
  sheet). Window sills and heads, 6'8" doors and 1 ft floor framing are typical
  builder values, not measured.
- **Elevation R front:** the builder rendering. Two-story stone entry tower with a
  front gable, arched porch opening, arched tower window, brick front with
  shutters and a soldier arch over the study window, brick garage gable with an
  arch over the 16 x 7 raised-panel door, siding on the sides and back
  (front-only masonry), hip roofs.
- **Options chosen (configuration sheet and change orders):** owner's bath
  Shower #2 (shower in place of the linen closet, oval tub), linen wall cabinet,
  drawer bank, kitchen island, 36 in uppers, gas range, whole-house gutters, coach
  lights, study in lieu of flex.
- **Finishes:** Irish Cream exterior package (SW 3542 Charwood, 7019 Gauntlet
  Grey, 7038 Tony Taupe, 7526 Maison Blanche), Buckskin Cream Chopped stone with
  white mortar, Albero Ramo 8x24 wood-look tile downstairs and in baths and
  laundry, Oyster carpet in bedroom 4, Fairfield Nutmeg cabinets, Caledonia
  granite. Upstairs is Lifeproof Sterling Oak vinyl plank, inferred from the 2021
  install quote (wood subfloor, stair nosing). Which paint goes on which surface
  is a guess.
- **Site:** the survey's lot lines and curve, setbacks, driveway, walk, rear
  concrete porch slab and AC pad.
- **Furniture layer:** the pieces named in the purchase receipts (sofa set,
  cocktail tables, oak dining set, charcoal sectional with ottoman). Placement is
  a guess.

## Known gaps

- The survey labels the rear slab "covered conc. porch", but no covered patio
  option is on the selections. It is modeled as an uncovered slab.
- Fence lines are approximate. The survey only marks fence at the lot lines.
- Roof pitch is assumed 6/12 for the main roof and garage and 9/12 for the tower.
- The lot is modeled flat; the survey has no spot elevations.
- Solar panels are not modeled; panel count and layout are not in these documents.
- Interior doors are openings with headers; there are no door leaves.
