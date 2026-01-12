Install packages to run:
pip install pandas matplotlib
pip install networkx
pip install geopandas geodatasets shapely pyproj fiona
pip install openpyxl

"""
Data_cleaning.py

What this script does: 
Data cleaning for Table.xlsx


Implements rules:
1) Metadata consistency check (flag)
2) Metadata contradiction fix
3) Study time window normalization to YYYY or YYYY-YYYY (and flag non-normalizable)
4) Keep only Conclusion in {Include, Maybe}
5) If DS Methods=None -> Computer vision task(s)=None
6) If CV task(s) != None -> DS Methods != None (flag)
7) If Dataset size (images) <100 AND DS Methods != None (flag)
8) Dataset size (images) missing -> Not reported
9) If Dataset size >=100k AND Data collection mode=Manual (flag)

Outputs:
- Table_cleaned.xlsx
- Table_cleaned.csv
- Table_flags.xlsx (rows needing review)
"""





"""
Analysis.py  (figures + printed results/conclusion)



Replicates Fig.2 and Fig.3 from the original study.

INPUT (expected in the same folder as this script):
   - Table_cleaned.xlsx  OR  Table_cleaned.csv

OUTPUT (saved in the same folder):
   - Fig2_geographic_distribution.png
   - Fig3_cross_continent_collaboration.png

Notes:
 - Fig2: colors countries by continent-level counts from "Study scale / case geography"
 - Fig3: bubble network of "Author affiliation continent(s)" with edge weights:
         solid = collaborations between exactly 2 continents


"""


"""
Tablesta.py  (Data statistics)



Statistical analysis of data for each research feature from the cleaned table.

INPUT (expected in the same folder as this script):
   - Table_cleaned.xlsx  OR  Table_cleaned.csv

OUTPUT (saved in the same folder):
   - *_counts.csv


"""


"""
25survey__Query_.pdf  (query code and reason)

"""