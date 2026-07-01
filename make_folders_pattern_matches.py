import make_SQL_database
import os

for e in set([e[0] for e in make_SQL_database.make_all_characters()]):
    os.makedirs(
        os.path.join("pattern_matches", e),
        exist_ok=True
    )
    os.makedirs(
        os.path.join("pattern_matches", e,"real"),
        exist_ok=True
    )
    os.makedirs(
        os.path.join("pattern_matches", e,"portrait"),
        exist_ok=True
    )
