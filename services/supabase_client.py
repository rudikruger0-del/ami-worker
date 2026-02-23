import logging
import os

from supabase import Client, create_client

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client | None = None

service_role_present = bool(SUPABASE_SERVICE_ROLE_KEY)
logger.info("Supabase startup: service role key present=%s", service_role_present)

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    logger.warning(
        "Supabase client disabled: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing."
    )
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    logger.info(
        "Supabase client initialized using service role key (SUPABASE_SERVICE_ROLE_KEY)."
    )
