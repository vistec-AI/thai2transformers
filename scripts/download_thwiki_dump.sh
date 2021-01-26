VERSION=$1
OUTPUT_DIR=$2

mkdir -p $OUTPUT_DIR

echo "Download thwiki-20201120-pages-articles.xml.bz2"
curl https://dumps.wikimedia.org/thwiki/${VERSION}/thwiki-${VERSION}-pages-articles.xml.bz2 \
--output $OUTPUT_DIR/thwiki-${VERSION}-pages-articles.xml.bz2