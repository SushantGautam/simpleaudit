import unittest
from simpleaudit.utils import strip_thinking

class TestContextStripping(unittest.TestCase):
    def test_strip_thinking_basic(self):
        text = "<think>some internal thought</think>Actual response"
        self.assertEqual(strip_thinking(text), "Actual response")
        
    def test_strip_thinking_multiline(self):
        text = """<think>
        thought line 1
        thought line 2
        </think>
        Actual response line 1
        Actual response line 2"""
        expected = "Actual response line 1\n        Actual response line 2"
        self.assertEqual(strip_thinking(text), expected)
        
    def test_strip_thinking_case_insensitive(self):
        text = "<THINK>Capital thought</THINK>Response"
        self.assertEqual(strip_thinking(text), "Response")
        
    def test_strip_thinking_no_tags(self):
        text = "Just a regular response"
        self.assertEqual(strip_thinking(text), text)
        
    def test_strip_thinking_unclosed_tag(self):
        text = "<think>Ongoing thought without end tag"
        self.assertEqual(strip_thinking(text), "")
        
    def test_strip_thinking_multiple_blocks(self):
        text = "<think>T1</think>R1<think>T2</think>R2"
        self.assertEqual(strip_thinking(text), "R1R2")

if __name__ == "__main__":
    unittest.main()
