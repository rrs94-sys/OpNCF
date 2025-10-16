#!/usr/bin/env python3
"""
data_collector.py - V3.0 FINAL
Fixed API endpoints + DataFrame conversion + Bearer auth
"""

import os
import sys
import certifi
import logging
from typing import Iterable, Optional, Dict, Any

import pandas as pd
import cfbd

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cfbd-data-collector")

os.environ["SSL_CERT_FILE"] = certifi.where()

_VALID_SEASON_TYPES = {
    "regular", "postseason", "both", "allstar",
    "spring_regular", "spring_postseason"
}


def _normalize_season_type(st: Optional[str]) -> str:
    """Normalize season type to valid CFBD enum"""
    if not st:
        return "regular"
    st_norm = str(st).strip().lower()
    if st_norm not in _VALID_SEASON_TYPES:
        raise ValueError(f"Invalid season_type: {st}")
    return st_norm


def _to_df(records: Optional[Iterable]) -> pd.DataFrame:
    """Convert CFBD API response to DataFrame"""
    if not records:
        return pd.DataFrame()
    
    try:
        records = list(records)
    except TypeError:
        return pd.DataFrame()
    
    if len(records) == 0:
        return pd.DataFrame()
    
    first_record = records[0]
    
    # Method 1: to_dict()
    if hasattr(first_record, 'to_dict'):
        try:
            return pd.DataFrame([r.to_dict() for r in records])
        except Exception:
            pass
    
    # Method 2: __dict__
    if hasattr(first_record, '__dict__'):
        try:
            data = [{k: v for k, v in vars(r).items() if not k.startswith('_')} 
                    for r in records]
            return pd.DataFrame(data)
        except Exception:
            pass
    
    # Method 3: Already dict
    if isinstance(first_record, dict):
        return pd.DataFrame(records)
    
    # Fallback
    try:
        return pd.DataFrame(records)
    except Exception as e:
        log.error(f"DataFrame conversion failed: {e}")
        return pd.DataFrame()


def _patch_cfbd_config(config: cfbd.Configuration) -> None:
    """Patch config for older cfbd client versions"""
    import types
    
    if not hasattr(config, "select_header_accept"):
        def select_header_accept(self, accepts):
            return accepts[0] if accepts else None
        config.select_header_accept = types.MethodType(select_header_accept, config)
    
    if not hasattr(config, "select_header_content_type"):
        def select_header_content_type(self, content_types):
            return content_types[0] if content_types else "application/json"
        config.select_header_content_type = types.MethodType(select_header_content_type, config)


class DataCollector:
    """CFBD API wrapper with fixed authentication and endpoints"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: str = "https://api.collegefootballdata.com"):
        # Try multiple environment variables
        api_key = (
            api_key or 
            os.getenv("CFBD_API_KEY") or 
            os.getenv("CFBDAPIKEY") or
            os.getenv("CFBD_KEY") or 
            os.getenv("CFB_API_KEY")
        )
        
        if not api_key or len(api_key.strip()) < 8:
            raise SystemExit(
                "Missing CFBD API key. Set CFBD_API_KEY environment variable."
            )
        
        # Configure authentication
        configuration = cfbd.Configuration()
        configuration.host = base_url
        configuration.api_key = configuration.api_key or {}
        configuration.api_key_prefix = configuration.api_key_prefix or {}
        configuration.api_key['Authorization'] = api_key.strip()
        configuration.api_key_prefix['Authorization'] = 'Bearer'
        
        _patch_cfbd_config(configuration)
        
        # Create API client
        self.api_client = cfbd.ApiClient(configuration)
        
        # Ensure Authorization header is set
        if "Authorization" not in self.api_client.default_headers:
            self.api_client.set_default_header(
                "Authorization", 
                f"Bearer {api_key.strip()}"
            )
        
        # Initialize API instances
        self.games_api = cfbd.GamesApi(self.api_client)
        self.betting_api = cfbd.BettingApi(self.api_client)
        self.ratings_api = cfbd.RatingsApi(self.api_client)
        self.teams_api = cfbd.TeamsApi(self.api_client)
        
        self.call_count = 0
    
    def get_games(self, year: int, week: Optional[int] = None, 
                  season_type: str = "regular") -> pd.DataFrame:
        """Fetch games"""
        try:
            st = _normalize_season_type(season_type)
            games = self.games_api.get_games(year=year, week=week, season_type=st)
            self.call_count += 1
            df = _to_df(games)
            log.info(f"get_games({year}, {week}, {st}) -> {len(df)} rows")
            return df
        except Exception as e:
            log.error(f"Error fetching games: {e}")
            return pd.DataFrame()
    
    def get_betting_lines(self, year: int, week: Optional[int] = None, 
                         season_type: str = "regular") -> pd.DataFrame:
        """Fetch betting lines"""
        try:
            st = _normalize_season_type(season_type)
            lines = self.betting_api.get_lines(year=year, week=week, season_type=st)
            self.call_count += 1
            df = _to_df(lines)
            log.info(f"get_betting_lines({year}, {week}, {st}) -> {len(df)} rows")
            return df
        except Exception as e:
            log.error(f"Error fetching lines: {e}")
            return pd.DataFrame()
    
    def get_sp_ratings(self, year: int) -> pd.DataFrame:
        """Fetch SP+ ratings - FIXED endpoint"""
        try:
            ratings = self.ratings_api.get_sp(year=year)
            self.call_count += 1
            df = _to_df(ratings)
            log.info(f"get_sp_ratings({year}) -> {len(df)} rows")
            return df
        except Exception as e:
            log.error(f"SP+ error: {e}")
            return pd.DataFrame()
    
    def get_fpi_ratings(self, year: int) -> pd.DataFrame:
        """Fetch FPI ratings - FIXED endpoint"""
        try:
            ratings = self.ratings_api.get_fpi(year=year)
            self.call_count += 1
            df = _to_df(ratings)
            log.info(f"get_fpi_ratings({year}) -> {len(df)} rows")
            return df
        except Exception as e:
            log.error(f"FPI error: {e}")
            return pd.DataFrame()
    
    def get_talent_rankings(self, year: int) -> pd.DataFrame:
        """Fetch talent rankings"""
        try:
            talent = self.teams_api.get_talent(year=year)
            self.call_count += 1
            df = _to_df(talent)
            log.info(f"get_talent_rankings({year}) -> {len(df)} rows")
            return df
        except Exception as e:
            log.error(f"Talent error: {e}")
            return pd.DataFrame()
    
    def get_teams(self) -> pd.DataFrame:
        """Fetch all FBS teams"""
        try:
            teams = self.teams_api.get_fbs_teams()
            self.call_count += 1
            df = _to_df(teams)
            log.info(f"get_teams() -> {len(df)} rows")
            return df
        except Exception as e:
            log.error(f"Teams error: {e}")
            return pd.DataFrame()
    
    def build_conference_lookup(self) -> Dict[str, str]:
        """Build team -> conference mapping"""
        teams_df = self.get_teams()
        if teams_df.empty:
            return {}
        
        school_col = next((c for c in ['school', 'team'] if c in teams_df.columns), None)
        conf_col = next((c for c in ['conference'] if c in teams_df.columns), None)
        
        if not school_col or not conf_col:
            log.warning(f"Cannot build conference lookup. Columns: {teams_df.columns.tolist()}")
            return {}
        
        lookup = {}
        for _, row in teams_df[[school_col, conf_col]].dropna().iterrows():
            school = str(row[school_col]).strip()
            conf = str(row[conf_col]).strip()
            if school and conf:
                lookup[school] = conf
        
        log.info(f"Built conference lookup for {len(lookup)} teams")
        return lookup


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_collector.py <year> [week] [season_type]")
        sys.exit(0)
    
    year = int(sys.argv[1])
    week = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[2].strip() else None
    season_type = sys.argv[3] if len(sys.argv) >= 4 else "regular"
    
    dc = DataCollector()
    
    print(f"\nTesting data collection for {year} week {week or 'all'}...")
    
    games_df = dc.get_games(year=year, week=week, season_type=season_type)
    lines_df = dc.get_betting_lines(year=year, week=week, season_type=season_type)
    sp_df = dc.get_sp_ratings(year=year)
    fpi_df = dc.get_fpi_ratings(year=year)
    talent_df = dc.get_talent_rankings(year=year)
    conf_lookup = dc.build_conference_lookup()
    
    print(f"\nResults:")
    print(f"  Games: {len(games_df)}")
    print(f"  Lines: {len(lines_df)}")
    print(f"  SP+: {len(sp_df)}")
    print(f"  FPI: {len(fpi_df)}")
    print(f"  Talent: {len(talent_df)}")
    print(f"  Conferences: {len(conf_lookup)}")
    print(f"  API calls: {dc.call_count}")
    
    if not games_df.empty:
        print(f"\nSample game columns: {games_df.columns.tolist()[:10]}")
