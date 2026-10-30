import unittest
import pandas as pd
from label_schema_v3 import normalize_keys


class LabelSchemaTests(unittest.TestCase):
    def test_ipv6_icmp_address_family(self):
        key=pd.Series(['fe80::1:0|ff02::16:0|1'])
        self.assertEqual(normalize_keys(key).iloc[0],'fe80::1:0|ff02::16:0|58')

    def test_ipv4_icmp_unchanged(self):
        key=pd.Series(['192.168.1.1:0|192.168.1.2:0|1'])
        self.assertEqual(normalize_keys(key).iloc[0],key.iloc[0])

    def test_ipv6_spelling_and_direction(self):
        a=pd.Series(['2001:0db8:0000:0000:0000:0000:0000:0001:9|2001:db8::2:8|6'])
        b=pd.Series(['2001:db8::2:8|2001:db8::1:9|6'])
        self.assertEqual(normalize_keys(a).iloc[0],normalize_keys(b).iloc[0])

    def test_mixed_address_families_rejected(self):
        with self.assertRaises(ValueError):
            normalize_keys(pd.Series(['192.168.1.1:8|fe80::1:9|6']))

    def test_missing_address_is_not_fabricated(self):
        key=pd.Series(['0:0|192.168.1.1:0|1'])
        self.assertEqual(normalize_keys(key).iloc[0],key.iloc[0])


if __name__=='__main__':
    unittest.main(verbosity=2)
