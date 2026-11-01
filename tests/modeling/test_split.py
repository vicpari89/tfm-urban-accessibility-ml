import numpy as np
import pandas as pd

from urban_accessibility_ml.modeling.split import (
    BARRIO_COL, CITY_COL, CLASS_COL, ID_COL, N_SPLITS, STREET_COL, split_cross_city, split_mixed, split_within_city,
)


def make_df():
    rng = np.random.default_rng(0)
    rows = []
    for city in ["A", "B"]:
        for s in range(40):  # 40 streets per city, 1-5 tramos each
            for _ in range(rng.integers(1, 6)):
                rows.append((city, f"{city}_calle_{s}", rng.integers(0, 3)))
    df = pd.DataFrame(rows, columns=[CITY_COL, STREET_COL, CLASS_COL])
    df[BARRIO_COL] = df[STREET_COL].str.split("_").str[-1].astype(int) // 5  # 8 barrios per city, same names in both
    df[ID_COL] = df[CITY_COL] + "_" + pd.Series(range(len(df))).astype(str)
    return df


class TestSplitWithinCity:
    def setup_method(self):
        self.df = make_df()
        self.city = "A"
        self.city_df = self.df[self.df[CITY_COL] == self.city]
        self.splits = split_within_city(self.df, self.city)

    def test_number_of_folds(self):
        assert len(self.splits) == N_SPLITS

    def test_each_tramo_in_train_xor_test(self):
        all_ids = set(self.city_df[ID_COL])
        for s in self.splits:
            train, test = set(s.train_ids), set(s.test_ids)
            assert not train & test  # never in both
            assert train | test == all_ids  # every tramo used

    def test_each_tramo_tested_exactly_once(self):
        tested = np.concatenate([s.test_ids for s in self.splits])
        assert sorted(tested) == sorted(self.city_df[ID_COL])

    def test_no_street_shared_between_train_and_test(self):
        street = self.city_df.set_index(ID_COL)[STREET_COL]
        for s in self.splits:
            assert not set(street[s.train_ids]) & set(street[s.test_ids])

    def test_reproducible_with_the_seed(self):
        again = split_within_city(self.df, self.city)
        for a, b in zip(self.splits, again):
            assert np.array_equal(a.train_ids, b.train_ids)
            assert np.array_equal(a.test_ids, b.test_ids)

    def test_every_test_fold_has_all_classes(self):
        cls = self.city_df.set_index(ID_COL)[CLASS_COL]
        for s in self.splits:
            assert set(cls[s.test_ids]) == set(self.city_df[CLASS_COL])

    def test_only_requested_city(self):
        ids = set(self.city_df[ID_COL])
        for s in self.splits:
            assert set(s.train_ids) | set(s.test_ids) <= ids


class TestSplitCrossCity:
    def setup_method(self):
        self.df = make_df()
        self.split = split_cross_city(self.df, "A", "B")[0]

    def test_train_is_all_train_city_and_test_is_all_test_city(self):
        ids = lambda c: set(self.df[self.df[CITY_COL] == c][ID_COL])
        assert set(self.split.train_ids) == ids("A")
        assert set(self.split.test_ids) == ids("B")

    def test_no_overlap(self):
        assert not set(self.split.train_ids) & set(self.split.test_ids)

    def test_reverse_direction_swaps_sets(self):
        rev = split_cross_city(self.df, "B", "A")[0]
        assert set(rev.train_ids) == set(self.split.test_ids)
        assert set(rev.test_ids) == set(self.split.train_ids)


class TestSplitMixed:
    def setup_method(self):
        self.df = make_df()
        self.splits = split_mixed(self.df)

    def test_number_of_folds(self):
        assert len(self.splits) == N_SPLITS

    def test_each_tramo_in_train_xor_test(self):
        all_ids = set(self.df[ID_COL])
        for s in self.splits:
            train, test = set(s.train_ids), set(s.test_ids)
            assert not train & test
            assert train | test == all_ids

    def test_each_tramo_tested_exactly_once(self):
        tested = np.concatenate([s.test_ids for s in self.splits])
        assert sorted(tested) == sorted(self.df[ID_COL])

    def test_reproducible_with_the_seed(self):
        again = split_mixed(self.df)
        for a, b in zip(self.splits, again):
            assert np.array_equal(a.train_ids, b.train_ids)
            assert np.array_equal(a.test_ids, b.test_ids)

    def test_every_test_fold_has_all_classes(self):
        cls = self.df.set_index(ID_COL)[CLASS_COL]
        for s in self.splits:
            assert set(cls[s.test_ids]) == set(self.df[CLASS_COL])

    def test_both_cities_in_train_and_test(self):
        city = self.df.set_index(ID_COL)[CITY_COL]
        for s in self.splits:
            assert set(city[s.train_ids]) == {"A", "B"}
            assert set(city[s.test_ids]) == {"A", "B"}

    def test_no_city_street_shared_between_train_and_test(self):
        key = (self.df[CITY_COL] + "_" + self.df[STREET_COL]).set_axis(self.df[ID_COL])
        for s in self.splits:
            assert not set(key[s.train_ids]) & set(key[s.test_ids])

    def test_same_street_name_in_two_cities_is_independent(self):
        # "Calle Mayor" exists in both cities: they must be different groups
        df = make_df()
        df[STREET_COL] = df[STREET_COL].str.split("_").str[-1]  # drop city prefix
        city = df.set_index(ID_COL)[CITY_COL]
        street = df.set_index(ID_COL)[STREET_COL]
        pairs_split = any(
            {(city[i], street[i]) for i in s.train_ids} & {(city[i], street[i]) for i in s.test_ids}
            for s in split_mixed(df)
        )
        assert not pairs_split
        # and at least one fold puts the same name in train for one city and test for the other
        assert any(
            {street[i] for i in s.train_ids} & {street[i] for i in s.test_ids}
            for s in split_mixed(df)
        )


class TestGroupByBarrio:
    def setup_method(self):
        self.df = make_df().set_index(ID_COL, drop=False)

    def test_within_city_no_barrio_shared_between_train_and_test(self):
        for s in split_within_city(self.df, "A", group_col=BARRIO_COL):
            assert not set(self.df.loc[s.train_ids, BARRIO_COL]) & set(self.df.loc[s.test_ids, BARRIO_COL])

    def test_mixed_no_city_barrio_shared_between_train_and_test(self):
        key = self.df[CITY_COL] + "_" + self.df[BARRIO_COL].astype(str)
        for s in split_mixed(self.df, group_col=BARRIO_COL):
            assert not set(key[s.train_ids]) & set(key[s.test_ids])
