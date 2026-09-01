"""City-scale image geo-localization for Türkiye from Mapillary street-level imagery.

Pipeline (mirrors "Scaling Image Geo-Localization to Continent Level", Lindenberger et al., NeurIPS 2025,
scaled down to one city):

  collect  -> S2-cell hierarchy -> proxy classification training -> prototype database (+ aerial codes)
           -> cosine retrieval over cells -> local refinement -> recall@{25..5000 m}
"""

__version__ = "0.1.0"
