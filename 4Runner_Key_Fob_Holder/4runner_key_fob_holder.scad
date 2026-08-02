// ============================================================
// 2022 Toyota 4Runner (5th gen) — Key Fob Holder
// Drop-in insert for the coin pocket on the center console
// (the small rubber-bottomed cubby next to the shifter).
// Holds the smart key fob (HYQ14FBA/FBB) standing upright.
//
// IMPORTANT: verify the two POCKET measurements below with
// calipers before printing — trim pockets vary slightly and
// these defaults are best estimates. Everything is parametric.
//
// Print: PETG or ABS/ASA recommended (PLA can warp in a hot
// car). 0.2mm layers, 3 walls, 15% infill, no supports.
// Print in the orientation the STL comes in (slot up).
// ============================================================

// ---------- Coin pocket (measure yours!) ----------
pocket_length   = 64;    // longer axis of the pocket opening, mm
pocket_width    = 36;    // shorter axis of the pocket opening, mm
insert_depth    = 35;    // how far the insert reaches into the pocket
fit_clearance   = 0.4;   // total clearance (both sides combined)
bottom_taper    = 1.2;   // inset at the very bottom, eases insertion
corner_radius   = 4;     // pocket corner rounding

// ---------- Fob slot (HYQ14FBA smart key ~72 x 36.5 x 12) ----------
slot_length     = 38.5;  // fob width + clearance
slot_width      = 14.5;  // fob thickness (incl. buttons) + clearance
slot_depth      = 40;    // how deep the fob sinks in (fob sticks out ~32)
slot_radius     = 3;     // slot corner rounding

// ---------- Top collar (sits on the pocket rim) ----------
lip             = 2.5;   // flange overhang past the pocket opening
collar_height   = 8;     // height of the collar above the rim

// ---------- Anti-rattle ribs ----------
rib_d           = 1.0;   // rib diameter; protrudes rib_d/2, crushes on insert
rib_count       = 3;     // ribs per long face

$fn = 64;
eps = 0.01;

body_l  = pocket_length - fit_clearance;
body_w  = pocket_width  - fit_clearance;
total_h = insert_depth + collar_height;
floor_t = total_h - slot_depth;
assert(floor_t >= 2, "slot_depth too deep — leave at least 2mm of floor");

// 2D rounded rectangle centered at origin
module rrect(l, w, r) {
    offset(r) square([l - 2*r, w - 2*r], center = true);
}

// rib x-positions, evenly spread over the middle 70% of the long face
function rib_x(i) = (rib_count == 1) ? 0
    : -0.35*body_l + i * (0.7*body_l / (rib_count - 1));

module part() {
    difference() {
        union() {
            // short lead-in chamfer at the bottom for easy entry
            hull() {
                linear_extrude(eps)
                    rrect(body_l - 2*bottom_taper, body_w - 2*bottom_taper, corner_radius);
                translate([0, 0, 3])
                    linear_extrude(eps)
                        rrect(body_l, body_w, corner_radius);
            }
            // straight-walled lower section
            translate([0, 0, 3])
                linear_extrude(insert_depth - 3)
                    rrect(body_l, body_w, corner_radius);
            // collar that overhangs the pocket rim
            translate([0, 0, insert_depth])
                linear_extrude(collar_height)
                    rrect(body_l + 2*lip, body_w + 2*lip, corner_radius + lip);
            // vertical crush ribs on both long faces
            for (side = [-1, 1], i = [0 : rib_count - 1])
                translate([rib_x(i), side * body_w/2, 5])
                    cylinder(h = insert_depth - 10, d = rib_d);
        }
        // fob slot
        translate([0, 0, total_h - slot_depth])
            linear_extrude(slot_depth + eps)
                rrect(slot_length, slot_width, slot_radius);
    }
}

part();
