# Grassland review of the class filters (2026-11)

The class filters in `configs/filters/` used to treat almost all grassland as background. In
grassland-dominated regions that left most visible fields unlabelled: a model trained on those
chips learns that hedge-bounded meadows are background. Ireland's crop-only filter kept 7% of the
declared area; Switzerland's kept 14%.

Each filter was reviewed by eye. For every source, four chips were rendered over that year's Q3
Sentinel-2 quarterly mosaic, centred on randomly sampled parcels, with the current filter's
fields in yellow, the grassland it excluded in cyan, and every other excluded class in grey. A
grassland class became field when its parcels read as **contained fields**: closed shapes
bounded by hedges, ditches, roads, walls or forest edges, the same kind of object as the arable
parcels next to them. Parcel counts and area shares were recorded but did not decide anything.

## The rule

**Field:** managed grassland. That means mown meadows, permanent and temporary pasture on
farmland, grass and grass-clover leys, and forage grasses.

**Background:** grassland whose parcels are not contained fields, and grass that is not a field
crop:

- alpine, summer, mountain and common grazing (Alm, Sömmerungsweiden, Hutweide, Bergmähder,
  mountain pasture)
- rough, pastoral, shrub, wooded and forest pasture (pastoral areas, dehesa, karstic pasture)
- semi-natural and nature-managed grassland, and habitat types
- litter meadows, riparian and buffer strips, field margins and EFA edges
- turf and sod, energy grasses (reed canary grass, Miscanthus, switchgrass), and cover or
  green-manure grasses
- grassland fallow, and grassland taken out of production

Grass grown for seed keeps whatever decision each filter already made for it.

## Decisions by filter

Shares are of declared area (`metrics:area`) in the reviewed year, or of parcels where a source
has no area column. "Grassland" is every class the old filter excluded whose name is grassland.
Only the managed part of it moved.

| Filter | Reviewed | Old filter keeps | Grassland | Decision | Kept background |
|---|---|---|---|---|---|
| austria | at_2025 | 39% | 57% | include managed meadows and pastures (636, 637, 642, 701, 710, 715–717) and arable sequences with forage grass (177, 663) | Almweide (992, 28% of area), Hutweide (707, 722, 723), Bergmähder (708), Streuwiese (704), Grünlandbrache (721) |
| be-flanders | be_vlg_2025 | 62% | 35% | include grassland (60), grass-herb (63), grass-legume (745) | pasture with trees (9823), natural grassland (9828, 9829), turf, seed grass, green cover |
| be-wallonia | be_wal_2022 | 50% | 50% | include Prairie et fourrage (6) | — |
| switzerland | ch_zh_2025, ch_be_2025 | 14–36% | 51–84% | include artificial, permanent, extensive and low-intensity meadows and home pastures (601, 611–613, 616, 617, 660, 697) | summer and common pastures (930, 933; 41% of Bern), forest pastures (618, 625), summering hay meadows (621–623, 935), litter meadows (851, 936), riverside meadows (635), biodiversity areas (693, 694), non-eligible grassland (698), pig and poultry pasture (602) |
| de-bb-nds-nrw | de_nds_2025 | 68% | 28% | include meadows, mowing pastures, pastures, Ackergras, Festwiese, alternating and permanent grassland, reseeded permanent grassland (444) | Weiden und Almen (453), montane meadows (465, 467), Streuwiesen (458), heath (492), Altgrasstreifen (93), set-aside and out of production (567, 592), energy grasses |
| de-by | de_by_2026 | 63% | 37% | include Mahd- und Weideland (1020) | — |
| de-mv | de_mv_2026 | 78% | 20% | include DGL, 32G, 34G | afforestation on grassland (AGL) |
| de-sh, de-sl | de_sh_2025, de_sl_2026 | 67%, 42% | 33%, 53% | include Dauergrünland | — |
| de-th | de_th_2026 | 78% | 23% | include GL | — |
| denmark | dk_2025 | 80% | 10% | include permanent, rotational and environmental grass and clover-grass (170, 247, 250–259, 272, 274, 276, 278, 279, 286, 305, 306, 350) and grass undersown after cereals (960–962) | borehole grass (248, 249), sod and rolled turf (269, 275), pig paddocks (270, 287), EFA edges (335, 336), converted land (326), henyard (488), energy grasses |
| estonia | ee_2024 | 72% | 27% | include grassland and forage grasses (0062, 0092, 0116, 0148, 0215, 0239, 0258) | rough grazings (0046) already background |
| finland | fi_2025 | 53% | 40% | include forage, pasture and permanent grass (6111, 6112, 6121, 6122, 6130, 6140, 6210, 6220, 9412) | natural and wooded pasture, forest pasture, nature-management and diversity fields, environmental and protection-zone grass (9805, 9810, 9812), reed canary grass |
| france | fr_2024 | 56% | 41% | include permanent pasture (PPH), long and short rotation meadows (PRL, PTR) and forage grasses incl. Festulolium (XFE) | pastoral areas (SPH, SPL, SIN, SNU), turf/seed grass (GRA), Miscanthus |
| croatia | hr_2024 | 79% | 20% | include meadow (310), pasture (320), permanent grassland (300) | karstic pasture (321) |
| latvia | lv_2025 | 64% | 27% | include permanent grassland (710) and sown grasses (713, 720) | reed canary grass, switchgrass |
| lithuania | lt_2025 | 70% | 27% | include pastures-meadows and perennial grasses | — |
| netherlands | nl_2025 | 43% | 53% | include permanent, temporary and agricultural natural grassland, and forage ryegrass | natural grassland with nature as main function or unspecified, field and fauna margins, turf, cover crops, energy grasses, seed and turf fescues and bluegrass (rood-, riet-zwenkgras, veldbeemdgras) |
| portugal | pt_2025 | 33% | 52% | include temporary meadows (142), permanent pastures (143), forage grasses | pasture with bushes (280), common land pasture (283), halfa grass (289) |
| sweden | se_2025 | 46% | 49% | include mowing and grazing on arable land (49, 50), field haymaking (57) and Westerwolds ryegrass (301) | pasture not on arable land (52), forest and mountain grazing, mosaic pasture, restoration areas, energy grass |
| slovakia | sk_2025 | 66% | 27% | include permanent grassland, grassed arable land and fodder grasses | semi-natural habitat types A–G |
| slovenia | si_2024 | 38% | 61% | include permanent grassland (204), grass on arable land (201), grass-clover (203) | green cover/turf (202), silvergrass (722) |
| it_tuscany | it_1_2019 | 61% | 16% | include multifloral meadow (336), multifloral pasture (65), pasture (899), forage grass and leys (1464, 46, 47, 359, 361, 369) | wooded pasture (1172, 1173), pasture with exposed rock (1905, 1906), natural and arid grassland (380, 382, 391, 460), traditional grazing (218), land withdrawn from production (1700) |
| czechia | cz_2025 | 94% | 3% | include permanent and temporary grassland, forage grasses and grass mixtures | energy grasses, protective strips, reed canary grass, lemon grass, turf and seed grasses (bentgrasses, sheep/hair/hard fescue, tufted hairgrass, wood bluegrass, rough meadow grass) |
| es_cat | es_cat_2024 | 90% | 2% | include pastures under 5 years (127) | — |
| es_uso_north (new) | es_ga_2025, es_cb_2025, es_pv_2025 | 6–43% | 57–95% | Galicia, Cantabria and País Vasco: include pasture (PS), mostly small bounded meadows | shrub pasture (PR), wood pasture (PA) |
| es_uso | es_cl_2025 | 32% | 66% | unchanged: PS here is open rangeland sprawling over scrub | PS, PR, PA |
| ireland | ie_2022 | 7% | 91% | grassland included earlier (e39db07) | reed canary grass, riparian buffers |

**Unchanged, nothing to decide:** bulgaria (98% kept, no grassland excluded), japan (pasture
already field), lithuania 2021 (`ec_lt`, no grassland in that edition), es_cn (1.5%: grazing on
non-agricultural land only).

**Spain national (`es`), reviewed separately:** the first attempt sampled before filtering,
so every sample landed on the Canaries. Re-sampled by region, permanent (62) and young (63)
pasture split by landscape:

- In Galicia, Asturias, Cantabria and País Vasco they are small meadows bounded by hedges and
  trees.
- In Castilla y León they are mostly large parcels over wooded and scrubby pasture.
- In Extremadura they are dehesa enclosures.
- In Andalucía they are drove-road and riverside strips or scrubland.
- In Soria and Guadalajara they outline slopes, scrub and forest.

`spain_filter.yaml` keeps them background, because one rule has to hold for the whole country.
The Spain 2025 union (`es_merged_filter.yaml`) counts them as field in the four Atlantic
regions only. That matches the regional `es_uso_north` decision.

**Galicia (`es_ga`), excluded:** a second review of the union's gap parcels (regional parcels
the national edition doesn't cover) found Galicia's to be mostly plots of a few hundred square
metres around villages. Even above 0.1 ha, many sit under tree cover with no open field
visible: registered plots that have been abandoned and grown over. The national edition
alone leaves about half of Galicia's farmland unlabelled, so neither source makes clean
chips there. Every `es_ga` year is excluded. Galicia's national 2025 parcels go to the
presence-only dataset `es_po_2025`, where unlabelled land counts as unknown, not background.

**Spain 2025 union, other regions:** the union adds gap parcels of at least 0.1 ha only in
País Vasco, where the national edition alone leaves about a quarter of the farmland
unlabelled and the gap parcels read as meadows. Cantabria's gap parcels looked as good, but
its SIGPAC data is not CC-BY, so they are left out to keep es_2025 under CC-BY-4.0.
Andalucía and Castilla y León get no gap parcels (they included strips along roads, canals
and railways and patches in scrub). The national parcels of Andalucía, Castilla y León and
Cantabria go to `es_po_2025` with Galicia's, since without gap parcels many visible fields
there have no label.

**Tuscany (`it_tuscany`), reviewed later:** the source file was missing at the first review.
Sample masks of 38 it_1 2019 chips, with the old and the wider filter side by side, showed
meadow and forage grass inside arable blocks, where the old mask left holes. Pasture (899, 65)
on plains and valley floors reads as fields; in the hills much of it is rough grazing between
woods and scrub. It counts as field everywhere, one rule for the region; the split of pasture
area between lowland and hill was not measured. The field share of 2019 goes from 61% to 76%.

## Reproducing

Both scripts live in the benchmark-data-catalog repo under `tools/review/`. The renders came
from `grass_review.py <dataset_id> <filter.yaml> <imagery_year> <out_dir>`, which reads the
harmonized source over HTTP and the mosaics from the Source Cooperative mirror. The class moves
were made by `grass_filters.py`, which matches grassland by name and applies per-filter
keep-out lists.
