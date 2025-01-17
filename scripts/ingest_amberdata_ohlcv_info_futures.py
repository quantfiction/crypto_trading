import logging
from crypto_trading.data.amberdata_ingestor import AmberdataOHLCVInfoIngestor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        ingestor = AmberdataOHLCVInfoIngestor()
        ingestor.ingest_ohlcv_info()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise
