import logging
from crypto_trading.data.amberdata_ingestor import AmberdataOHLCVIngestor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        ingestor = AmberdataOHLCVIngestor()
        ingestor.ingest_ohlcv_data()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise
