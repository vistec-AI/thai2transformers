## Thai Wikipedia dataset preparation

This page explains how to download, extract and filter and clean texts from Thai Wikipedia dump

### Instruction


1. Download Thai Wikipedia dump with the following script (`./scripts/download_thwiki_dump.sh`)

    ```
    bash ./scripts/download_thwiki_dump.sh \
    20200820 \
    ./data/dataset/thwiki-20200820/1_dumps/
    ```

    <details>
    <summary>Example output:</summary>

    ```
    Download thwiki-20201120-pages-articles.xml.bz2
    % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                    Dload  Upload   Total   Spent    Left  Speed
    100  276M  100  276M    0     0  1763k      0  0:02:40  0:02:40 --:--:-- 4010k
    ```
    </details>

    <br>
    
    where `20200820` is the version of Thai Wikipedia dump (see more detail from [dumps.wikimedia.org/thwiki/20200820](https://dumps.wikimedia.org/thwiki/20200820))

    __Filename:__ thwiki-20201120-pages-articles.xml.bz2
    
    - SHA1: __1de130e11aa66c89b9ab0c73b5b6e739f423205b__

<br>

2. Extract texts from downloaded dump (.bz2) 

    2.1 Install `wikiextractor` a tool to extract text segments from dump file.

    ```bash
    bash ./scripts/install_wikiextractor.sh
    ```

    2.2 Install faketime

    ```bash
    apt-get update
    apt-get install faketime
    ```

    2.3  Extract texts from downloaded dump (.bz2) with `wikiextractor` via the following scripts (`./scripts/extract_thwiki_dump.sh`)
    
    ```bash
    faketime '2020-08-25 12:00:00' bash ./scripts/extract_thwiki_dump.sh \
    ./data/dataset/thwiki-20200820/1_dumps/thwiki-20200820-pages-articles.xml.bz2 \
    data/dataset/thwiki-20200820/2_extracted \
    logs/wikiextractor_thwiki-20200820-nolist \
    "--json --sections"
    ```

    where the arguments are as follows:

    1. DUMP_FILE_PATH - The path to the Wikipedia dump (.bz2)

    2. OUTPUT_DIR - Directory to store the extracted data

    3. LOG_PATH - Path to store the logging from wikiextractor

    4. PARAMS - Additina parameters that will be passed to `wikiextractor` (e.g. `--sections --json`) (See more detail from this page: https://github.com/attardi/wikiextractor)

    <br>

    <details>
    <summary>Example output:</summary>

    ```
    Begin extracting thwiki dump from ./data/dataset/thwiki-20200820/1_dumps/thwiki-20200820-pages-articles.xml.bz2
    INFO: Loaded 0 templates in 0.0s
    INFO: Starting page extraction from ./data/dataset/thwiki-20200820/1_dumps/thwiki-20200820-pages-articles.xml.bz2.
    INFO: Using 1 extract processes.
    INFO: 1	หน้าหลัก
    INFO: 545	ดาราศาสตร์
    INFO: 547	ภูมิศาสตร์
    INFO: 611	พันทิป.คอม
    INFO: 613	พันธุ์ทิพย์พลาซ่า
    INFO: 615	วิทยาการคอมพิวเตอร์
    INFO: 616	คณิตศาสตร์
    INFO: 618	การประมวลสารสนเทศ
    INFO: 619	การเมือง
  
    ...
    ...

    INFO: 1119875	ประเทศไอซ์แลนด์ในโอลิมปิกเยาวชนฤดูร้อน 2014
    INFO: 1119877	ประเทศอินโดนีเซียในโอลิมปิกเยาวชนฤดูร้อน 2014
    INFO: 1119879	ประเทศอิรักในโอลิมปิกเยาวชนฤดูร้อน 2014
    INFO: 1119880	ประเทศลัตเวียในโอลิมปิกเยาวชนฤดูร้อน 2014
    INFO: 1119881	ประเทศโมร็อกโกในโอลิมปิกเยาวชนฤดูร้อน 2014
    INFO: 1119882	ทีมผสมในโอลิมปิกเยาวชนฤดูร้อน 2014
    INFO: 1119883	ผลกระทบกิบส์–ดอนนัน
    INFO: Finished 79-process extraction of 139744 articles in 252.0s (554.6 art/s)
    INFO: total of page: 264219, total of articl page: 139744; total of used articl page: 139744

    ```
    </details>

<br>

3. Preprocess and aggregated into one file (.txt) with `preprocess_thwiki_extracted.py`
    
    The script will perform text preprocessing as follows:

    -  Remove article title if it is duplicated in the  first paragraph of the article

    - (Optional) Remove first empty parenthesis

    - (Optional) Split long segments (both Thai, and English)

    - (Optional) Add end of document token

    - (Optional) Replace space token with a special token `"<_>"`


    ```bash
    python ./scripts/preprocess_thwiki_extracted.py \
    ./data/dataset/thwiki-20200820/2_extracted \
    ./data/dataset/thwiki-20200820/3_aggregated \
    --remove_first_empty_parenthesis \
    --split_long_segment \
    --add_end_of_doc_token \
    --space_token "<_>" 
    ```

    <details>
    <summary>Example output:</summary>

    ```
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
    Begin loading files from ./data/dataset/thwiki-20200820/2_extracted
    Sub directory: ./data/dataset/thwiki-20200820/2_extracted/AE
    Sub directory: ./data/dataset/thwiki-20200820/2_extracted/AC
    Sub directory: ./data/dataset/thwiki-20200820/2_extracted/AB
    Sub directory: ./data/dataset/thwiki-20200820/2_extracted/AA
    Sub directory: ./data/dataset/thwiki-20200820/2_extracted/AD
    Sub directory: ./data/dataset/thwiki-20200820/2_extracted/AF
    Total number of files: 586
    Done.

    Begin extracting data
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 586/586 [00:05<00:00, 102.66it/s]
    139744it [00:02, 47093.23it/s]
    Done.

    Argumnet: remove_first_empty_parenthesis = True, Begin removing first empty parenthesis.
    139744it [00:00, 173145.59it/s]
    Done.

    Argumnet: split_long_segment = True, Begin spliting long segment.
    139744it [15:22, 151.55it/s]
    Done.

    Argumnet: add_end_of_doc_token = True, Begin adding end of document token `</s></s>`.
    139744it [00:00, 500688.78it/s]
    Done.

    Begin replaceing space with space token
    Argument space_token = <_>
    ```

    </details>

    <br>

    The following script will write the output to this path: `./data/dataset/thwiki-20200820/3_aggregated/thwiki.txt`

    SHA1 of the output file, `thwiki.txt`: __8c32b81bca7256816f359bda0531262d9c1f825a__
    
4. Clean data with the following script `clean_data-thwiki.py`

    This script will apply two text cleaning rules:

    1.  Replace non-breaking space with a space token 
    
    2.  Remove soft-hyphen and zero-width non-breaking space (invisible characters)

    ```bash
    python ./scripts/clean_data-thwiki.py \
    ./data/dataset/thwiki-20200820/3_aggregated/thwiki.txt \
    ./data/dataset/thwiki-20200820/4_cleaned/thwiki.txt
    ```

    <details>
    <summary>Example output:</summary>

    ```
    Begin reading file from ./data/dataset/thwiki-20200820/3_aggregated/thwiki.txt
    Done.

    Apply text cleaning rule 1: Replace non-breaking space with space token.
    Done.

    Apply text cleaning rule 2: Remove invisible characters.
    Done.

    Begin writing file to ./data/dataset/thwiki-20200820/4_cleaned/thwiki.txt
    ```

    </details>

5. Split into train/val/test set via the script `split_data.py`


    ```bash
    python ./scripts/split_data.py \
    ./data/dataset/thwiki-20200820/4_cleaned/thwiki.txt \
    ./data/dataset/thwiki-20200820/5_split
    ```


    <details>
    <summary>Example output:</summary>

    ```
    INFO: Load text file from ./data/dataset/thwiki-20200820/4_cleaned/thwiki.txt
    INFO: Begin splitting data.
        train_ratio: 0.95
        val_ratio: 0.025
        test_ratio: 0.025

    INFO: Train/val/test statistics.
        train set: 944782
        val set: 24863
        test set: 24862

    INFO: Begin writing train split to "./data/dataset/thwiki-20200820/5_split/train/train.txt".
    INFO: Begin writing val split to "./data/dataset/thwiki-20200820/5_split/val/val.txt".
    INFO: Begin writing test split to "./data/dataset/thwiki-20200820/5_split/test/test.txt".

    INFO: Done writing all split.

    ```

    </details>

    <br>

    SHA1 for each file:

    ```
    7fd01c8b5e90f4452ecdde1f92a75094e6187a78  test/test.txt
    f4e472ecbea284ffd6ebb3766636e8508c8cfc10  train/train.txt
    a472d1d7fa292f6e6b1f29e0afc64e94414bac44  val/val.txt
    ```