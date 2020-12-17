## Thai Wikipedia dataset preparation

This page explains how to download, extract and filter and clean texts from Thai Wikipedia dump

### Instruction


1. Download Thai Wikipedia dump with the following script (`./scripts/download_thwiki_dump.sh`)


    <br>

    ```bash
    cd scripts
    bash download_thwiki_dump.sh 20201120
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
    
    where `20201120` is the version of Thai Wikipedia dump (see more detail from [dumps.wikimedia.org/thwiki](https://dumps.wikimedia.org/thwiki/))

<br>

2. Extract texts from downloaded dump (.bz2) with `wikiextractor` via the following scripts (`./scripts/extract_thwiki_dump.sh`)

    ```
    cd scripts

    bash extract_thwiki_dump.sh \
    ../data/dataset/thwiki-20201120/dumps/thwiki-20201120-pages-articles.xml.bz2 \
    ../data/dataset/thwiki-20201120/extracted/thwiki-20201120_no-list/ \
    ../logs/wiki_extractor/thwiki-20201120_no-list \
    "--json --sections --filter_disambig_pages --discard_elements gallery,timeline,noinclude,pre,table,tr,td,th,caption,div,form,input,select,option,textarea,ul,li,ol,dl,dt,dd,menu,dir,ref,references,img,imagemap,source,small,br"
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
    Begin extracting thwiki dump from ../dataset/thwiki-20201120/dumps/thwiki-20201120-pages-articles.xml.bz2.
    INFO: Loaded 0 templates in 0.0s
    INFO: Starting page extraction from ../dataset/thwiki-20201120/dumps/thwiki-20201120-pages-articles.xml.bz2.
    INFO: Using 11 extract processes.
    INFO: 1	หน้าหลัก
    INFO: 545	ดาราศาสตร์
    INFO: 547	ภูมิศาสตร์
    INFO: 611	พันทิป.คอม
    INFO: 613	พันธุ์ทิพย์พลาซ่า
    INFO: 615	วิทยาการคอมพิวเตอร์
    INFO: 618	การประมวลสารสนเทศ
    INFO: 616	คณิตศาสตร์
    INFO: 619	การเมือง
    INFO: 660	ดิมมูบอร์เกียร์
    INFO: 662	เกษตรศาสตร์
    ...
    ...
    ...
    INFO: 1133008	อินเดอะมูดฟอร์เลิฟ
    INFO: 1133017	ถ้ำเอลโลรา
    INFO: 1133026	ซีเอฟเอ็นเอ็ม
    INFO: 1133035	เฮอริเคนไอโอตา
    INFO: 1133037	เฮอริเคนอีตา
    INFO: 1133038	ปลาสเตอร์เจียนเปอร์เซีย
    INFO: 1133051	มานาซูรุ
    INFO: Finished 11-process extraction of 140545 articles in 170.3s (825.5 art/s)
    INFO: total of page: 265524, total of articl page: 140604; total of used articl page: 140545

    ```
    </details>

<br>

3. Aggregate data

    Sourc: `../data/dataset/thwiki-20201120/extracted/thwiki-20201120_no-list/`


    ```
    ```

4. Data preprocessing. This preprocessing script includes functions as follows:

    - Text cleaning
    - Text filtering
    - English and Thai sentence breaking

    ```
    ```


5. Split train/val/test set


