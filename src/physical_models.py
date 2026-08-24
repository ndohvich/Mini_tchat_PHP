"""Optional pvlib clear-sky model executed only with supplied site coordinates."""
import pandas as pd

def physical_clear_sky(datetimes,config):
    """Use pvlib Ineichen only when latitude/longitude are configured, never inferred."""
    site=config.get("site",{}); lat,lon=site.get("latitude"),site.get("longitude")
    if lat is None or lon is None: return None,"Modèle physique non exécuté : variables physiques requises indisponibles."
    try:
        import pvlib
        loc=pvlib.location.Location(lat,lon,altitude=site.get("altitude") or 0); return loc.get_clearsky(pd.DatetimeIndex(datetimes))["ghi"].to_numpy(),None
    except ImportError: return None,"Physical model skipped: optional dependency pvlib is not installed."
