DUMP_FILE_PATH=$1
OUTPUT_DIR=$2
LOG_PATH=$3
PARAMS=$4

echo "Begin extracting thwiki dump from $DUMP_FILE_PATH"

python3 -m wikiextractor.WikiExtractor \
--output $OUTPUT_DIR \
--log_file $LOG_PATH \
$PARAMS \
$DUMP_FILE_PATH 