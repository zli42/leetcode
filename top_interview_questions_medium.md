# Top Interview Questions Medium Collection

## Array and Strings

### [3Sum](https://leetcode.cn/problems/3sum/)

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        if n < 3:
            return []
        
        nums.sort()
        
        res = []
        for i in range(n):
            if nums[i] > 0:
                return res
            
            if i > 0 and nums[i] == nums[i - 1]:
                continue
                
            l = i + 1
            r = n - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s == 0:
                    res.append([nums[i], nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1
                elif s < 0:
                    l += 1
                else:
                    r -= 1
        return res
```

* 时间复杂度：$O(N^2)$，其中 $N$ 是数组 $nums$ 的长度。
* 空间复杂度：$O(logN)$。我们忽略存储答案的空间，额外的排序的空间复杂度为 $O(logN)$。然而我们修改了输入的数组 $nums$，在实际情况下不一定允许，因此也可以看成使用了一个额外的数组存储了 $nums$ 的副本并进行排序，空间复杂度为 $O(N)$。

### [Set Matrix Zeroes](https://leetcode.cn/problems/set-matrix-zeroes/)

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        first_col = False
        first_row = False
        
        for i in range(m):
            if matrix[i][0] == 0:
                first_col = True
                break
                
        for j in range(n):
            if matrix[0][j] == 0:
                first_row = True
                break
                
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
                    
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
                    
        if first_col:
            for i in range(m):
                matrix[i][0] = 0
            
        if first_row:
            for j in range(n):
                matrix[0][j] = 0
```

* 时间复杂度：$O(mn)$，其中 $m$ 是矩阵的行数，$n$ 是矩阵的列数。我们至多只需要遍历该矩阵两次。
* 空间复杂度：$O(1)$。我们只需要常数空间存储若干变量。

### [Group Anagrams](https://leetcode.cn/problems/group-anagrams/)

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = dict()
        for item in strs:
            key = ''.join(sorted(item))
            res[key] = res.get(key, []) + [item]
        return list(res.values())
```

* 时间复杂度：$O(nk \log k)$，其中 $n$ 是 `strs` 中的字符串的数量，$k$ 是 `strs` 中的字符串的的最大长度。需要遍历 $n$ 个字符串，对于每个字符串，需要 $O(k \log k)$ 的时间进行排序以及 $O(1)$ 的时间更新哈希表，因此总时间复杂度是 $O(nk \log k)$。
* 空间复杂度：$O(nk)$，其中 $n$ 是 `strs` 中的字符串的数量，$k$ 是 $strs$ 中的字符串的的最大长度。需要用哈希表存储全部字符串。

### [Longest Substring Without Repeating Characters](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        if n < 2:
            return n
        
        t = set(s[0])
        right = 1
        res = 0
        for left in range(n):
            while right < n and s[right] not in t:
                t.add(s[right])
                right += 1
            t.remove(s[left])
            res = max(res, right - left)
        return res
```

* 时间复杂度：$O(N)$，其中 $N$ 是字符串的长度。左指针和右指针分别会遍历整个字符串一次。
* 空间复杂度：$O(|\Sigma|)$，其中 $\Sigma$ 表示字符集（即字符串中可以出现的字符），$|\Sigma|$ 表示字符集的大小。在本题中没有明确说明字符集，因此可以默认为所有 ASCII 码在 $[0, 128)$ 内的字符，即 $|\Sigma| = 128$。我们需要用到哈希集合来存储出现过的字符，而字符最多有 $|\Sigma|$ 个，因此空间复杂度为 $O(|\Sigma|)$。

### [Longest Palindromic Substring](https://leetcode.cn/problems/longest-palindromic-substring/)

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        def expandAroundCenter(left, right):
            while left >= 0 and right < n and s[left] == s[right]:
                left -= 1
                right += 1
            return left + 1, right - 1
        
        start = 0
        end = 0
        for i in range(n):
            left1, right1 = expandAroundCenter(i, i)
            if right1 - left1 > end - start:
                start = left1
                end = right1
            left2, right2 = expandAroundCenter(i, i+1)
            if right2 - left2 > end - start:
                start = left2
                end = right2
        return s[start:end+1]
```

* 时间复杂度：$O(n^2)$，其中 $n$ 是字符串的长度。长度为 $1$ 和 $2$ 的回文中心分别有 $n$ 和 $n-1$ 个，每个回文中心最多会向外扩展 $O(n)$ 次。
* 空间复杂度：$O(1)$。


## Tree and Graphs

## Sorting and Searching

## Dynamic Programming