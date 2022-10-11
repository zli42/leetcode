### [Reverse String](https://leetcode.cn/problems/reverse-string/)

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        i = 0
        j = len(s) - 1
        while i < j:
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1
```

* 时间复杂度：$O(N)$，其中 $N$ 为字符数组的长度。一共执行了 $N/2$ 次的交换。
* 空间复杂度：$O(1)$。只使用了常数空间来存放若干变量。

### [Reverse Integer](https://leetcode.cn/problems/reverse-integer/)

```python
class Solution:
    def reverse(self, x: int) -> int:
        INT_MIN, INT_MAX = -2**31, 2**31 - 1

        rev = 0
        while x != 0:
            # INT_MIN 也是一个负数，不能写成 rev < INT_MIN // 10
            if rev < INT_MIN // 10 + 1 or rev > INT_MAX // 10:
                return 0
            digit = x % 10
            # Python3 的取模运算在 x 为负数时也会返回 [0, 9) 以内的结果，因此这里需要进行特殊判断
            if x < 0 and digit > 0:
                digit -= 10

            # 同理，Python3 的整数除法在 x 为负数时会向下（更小的负数）取整，因此不能写成 x //= 10
            x = (x - digit) // 10
            rev = rev * 10 + digit
        
        return rev
```

* 时间复杂度：$O(log∣x∣)$。翻转的次数即 $x$ 十进制的位数。
* 空间复杂度：$O(1)$。

### [First Unique Character in a String](https://leetcode.cn/problems/first-unique-character-in-a-string/)

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        position = dict()
        for i, c in enumerate(s):
            if c in position:
                position[c] = -1
            else:
                position[c] = i
        n = len(s)
        first = n
        for p in position.values():
            if p != -1 and p < first:
                first = p
        if first == n:
            first = -1
        return first
```

* 时间复杂度：$O(n)$，其中 $n$ 是字符串 $s$ 的长度。第一次遍历字符串的时间复杂度为 $O(n)$，第二次遍历哈希映射的时间复杂度为 $O(|\Sigma|)$，由于 $s$ 包含的字符种类数一定小于 $s$ 的长度，因此 $O(|\Sigma|)$ 在渐进意义下小于 $O(n)$，可以忽略。
* 空间复杂度：$O(|\Sigma|)$，其中 $\Sigma$ 是字符集，在本题中 $s$ 只包含小写字母，因此 $|\Sigma| \leq 26$。我们需要 $O(|\Sigma|)$ 的空间存储哈希映射。

### [Valid Anagram](https://leetcode.cn/problems/valid-anagram/)

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        
        counter = dict()
        for c in s:
            counter[c] = counter.get(c, 0) + 1
        for c in t:
            counter[c] = counter.get(c, 0) - 1
            if counter[c] < 0:
                return False
        return True
```

* 时间复杂度：$O(n)$，其中 $n$ 为 $s$ 的长度。
* 空间复杂度：$O(S)$，其中 $S$ 为字符集大小，此处 $S=26$。

### [Valid Palindrome](https://leetcode.cn/problems/valid-palindrome/)

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        n = len(s)
        left, right = 0, n - 1
        
        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            if left < right:
                if s[left].lower() != s[right].lower():
                    return False
                left, right = left + 1, right - 1

        return True
```

* 时间复杂度：$O(|s|)$，其中 $|s|$ 是字符串 $s$ 的长度。
* 空间复杂度：$O(1)$。

### [String to Integer (atoi)](https://leetcode.cn/problems/string-to-integer-atoi/)

```python
class Automaton:
    def __init__(self):
        self.INT_MAX = 2**31 - 1
        self.INT_MIN = -(2**31)
        self.sign = 1
        self.ans = 0
        self.state = "start"
        self.table = {
            "start": ["start", "signed", "in_number", "end"],
            "signed": ["end", "end", "in_number", "end"],
            "in_number": ["end", "end", "in_number", "end"],
            "end": ["end", "end", "end", "end"],
        }

    def get_col(self, c):
        if c.isspace():
            return 0
        if c == "+" or c == "-":
            return 1
        if c.isdigit():
            return 2
        return 3

    def get(self, c):
        self.state = self.table[self.state][self.get_col(c)]
        if self.state == "in_number":
            self.ans = self.ans * 10 + int(c)
            self.ans = (
                min(self.ans, self.INT_MAX)
                if self.sign == 1
                else min(self.ans, -self.INT_MIN)
            )
        elif self.state == "signed":
            self.sign = 1 if c == "+" else -1


class Solution:
    def myAtoi(self, str: str) -> int:
        automaton = Automaton()
        for c in str:
            automaton.get(c)
        return automaton.sign * automaton.ans
```

* 时间复杂度：$O(n)$，其中 $n$ 为字符串的长度。我们只需要依次处理所有的字符，处理每个字符需要的时间为 $O(1)$。
* 空间复杂度：$O(1)$。自动机的状态只需要常数空间存储。

### [Implement strStr()](https://leetcode.cn/problems/implement-strstr/)

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        n = len(haystack)
        m = len(needle)
        if m == 0:
            return 0
        
        next_array = [0 for _ in range(m)]
        j = 0
        for i in range(1, m):
            while j > 0 and needle[i] != needle[j]:
                j = next_array[j - 1]
            if needle[i] == needle[j]:
                j += 1
            next_array[i] = j
            
        j = 0
        for i in range(n):
            while j > 0 and haystack[i] != needle[j]:
                j = next_array[j - 1]
            if haystack[i] == needle[j]:
                j += 1
            if j == m:
                return i - m + 1
        return -1
```

* 时间复杂度：$O(n+m)$，其中 $n$ 是字符串 `haystack` 的长度，$m$ 是字符串 `needle` 的长度。我们至多需要遍历两字符串一次。
* 空间复杂度：$O(m)$，其中 $m$ 是字符串 `needle` 的长度。我们只需要保存字符串 `needle` 的前缀函数。

### [Longest Common Prefix](https://leetcode.cn/problems/longest-common-prefix/)

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        prefix = strs[0]
        for i in range(1, len(strs)):
            j = 0
            while j < min(len(prefix), len(strs[i])) and prefix[j] == strs[i][j]:
                j += 1
            prefix = prefix[:j]
            if j == 0:
                break
        return prefix
```

* 时间复杂度：$O(mn)$，其中 $m$ 是字符串数组中的字符串的平均长度，$n$ 是字符串的数量。最坏情况下，字符串数组中的每个字符串的每个字符都会被比较一次。
* 空间复杂度：$O(1)$。使用的额外空间复杂度为常数。
