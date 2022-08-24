# Top Interview Questions Easy Collection

## Array

### [Remove Duplicates from Sorted Array](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        j = 0
        for i in range(len(nums)):
            if nums[i] != nums[j]:
                j += 1
                nums[j] = nums[i]
        return j + 1
```

* 时间复杂度：$O(n)$，其中 $n$ 是数组的长度。快指针和慢指针最多各移动 $n$ 次。
* 空间复杂度：$O(1)$。只需要使用常数的额外空间。

### [Best Time to Buy and Sell Stock II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i - 1]
            if diff > 0:
                profit += diff
        return profit
```

* 时间复杂度：$O(n)$，其中 $n$ 为数组的长度。我们只需要遍历一次数组即可。
* 空间复杂度：$O(1)$。只需要常数空间存放若干变量。

### [Rotate Array](https://leetcode.cn/problems/rotate-array/)

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k %= n
        self.reverse(nums, 0, n - 1)
        self.reverse(nums, 0, k - 1)
        self.reverse(nums, k, n - 1)

    def reverse(self, nums, start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
```

* 时间复杂度：$O(n)$，其中 $n$ 为数组的长度。每个元素被翻转两次，一共 $n$ 个元素，因此总时间复杂度为 $O(2n)=O(n)$。
* 空间复杂度：$O(1)$。

### [Contains Duplicate](https://leetcode.cn/problems/contains-duplicate/)

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        s = set()
        for num in nums:
            if num in s:
                return True
            s.add(num)
        return False
```

* 时间复杂度：$O(N)$，其中 $N$ 为数组的长度。
* 空间复杂度：$O(N)$，其中 $N$ 为数组的长度。

### [Single Number](https://leetcode.cn/problems/single-number/)

位异或运算
1. 任何数和 $0$ 做异或运算，结果仍然是原来的数，即 $a \oplus 0=a$。
2. 任何数和其自身做异或运算，结果是 $0$，即 $a \oplus a=0$。
3. 异或运算满足交换律和结合律，即 $a \oplus b \oplus a=b \oplus a \oplus a=b \oplus (a \oplus a)=b \oplus0=b$。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res ^= num
        return res
```

* 时间复杂度：$O(n)$，其中 $n$ 是数组长度。只需要对数组遍历一次。
* 空间复杂度：$O(1)$。

### [Intersection of Two Arrays II](https://leetcode.cn/problems/intersection-of-two-arrays-ii/)

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if len(nums1) > len(nums2):
            return self.intersect(nums2, nums1)

        m = dict()
        for num in nums1:
            m[num] = m.setdefault(num, 0) + 1

        intersection = list()
        for num in nums2:
            if m.get(num, 0) > 0:
                intersection.append(num)
                m[num] -= 1

        return intersection
```

* 时间复杂度：$O(m+n)$，其中 $m$ 和 $n$ 分别是两个数组的长度。需要遍历两个数组并对哈希表进行操作，哈希表操作的时间复杂度是 $O(1)$，因此总时间复杂度与两个数组的长度和呈线性关系。
* 空间复杂度：$O(\min(m,n))$，其中 $m$ 和 $n$ 分别是两个数组的长度。对较短的数组进行哈希表的操作，哈希表的大小不会超过较短的数组的长度。为返回值创建一个数组 `intersection`，其长度为较短的数组的长度。


### [Plus One](https://leetcode.cn/problems/plus-one/)

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        n = len(digits)
        for i in range(n - 1, -1, -1):
            if digits[i] != 9:
                digits[i] += 1
                for j in range(i + 1, n):
                    digits[j] = 0
                return digits

        # digits 中所有的元素均为 9
        return [1] + [0] * n
```

* 时间复杂度：$O(n)$，其中 $n$ 是数组 `digits` 的长度。
* 空间复杂度：$O(1)$。返回值不计入空间复杂度。

### [Move Zeroes](https://leetcode.cn/problems/move-zeroes/)

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1
```

* 时间复杂度：$O(n)$，其中 $n$ 为序列长度。每个位置至多被遍历两次。
* 空间复杂度：$O(1)$。只需要常数的空间存放若干变量。

### [Two Sum](https://leetcode.cn/problems/two-sum/)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        res = dict()
        for i, num in enumerate(nums):
            if num in res:
                return [res[num], i]
            res[target - num] = i
        return list()
```

* 时间复杂度：$O(N)$，其中 $N$ 是数组中的元素数量。对于每一个元素 x，我们可以 $O(1)$ 地寻找 target - x。
* 空间复杂度：$O(N)$，其中 $N$ 是数组中的元素数量。主要为哈希表的开销。

### [Valid Sudoku](https://leetcode.cn/problems/valid-sudoku/)

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row = [set() for _ in range(9)]
        col = [set() for _ in range(9)]
        cell = [set() for _ in range(9)]
        for i in range(9):
            for j in range(9):
                n = board[i][j]
                if n == ".":
                    continue
                if n in row[i] or n in col[j] or n in cell[i // 3 * 3 + j // 3]:
                    return False
                row[i].add(n)
                col[j].add(n)
                cell[i // 3 * 3 + j // 3].add(n)
        return True
```

* 时间复杂度：$O(1)$。数独共有 $81$ 个单元格，只需要对每个单元格遍历一次即可。
* 空间复杂度：$O(1)$。由于数独的大小固定，因此哈希表的空间也是固定的。

### [Rotate Image](https://leetcode.cn/problems/rotate-image/)

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        # 水平翻转
        for i in range(n // 2):
            for j in range(n):
                matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
        # 主对角线翻转
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```

* 时间复杂度：$O(N^2)$，其中 $N$ 是 `matrix` 的边长。对于每一次翻转操作，我们都需要枚举矩阵中一半的元素。
* 空间复杂度：$O(1)$。为原地翻转得到的原地旋转。

## Strings

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

KMP


## Linked List

## Trees

### [Maximum Depth of Binary Tree](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

DFS

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root):
        if root is None: 
            return 0 
        else: 
            left_height = self.maxDepth(root.left) 
            right_height = self.maxDepth(root.right) 
            return max(left_height, right_height) + 1 
```

* 时间复杂度：$O(n)$，其中 $n$ 为二叉树节点的个数。每个节点在递归中只被遍历一次。
* 空间复杂度：$O(height)$，其中 `height` 表示二叉树的高度。递归函数需要栈空间，而栈空间取决于递归的深度，因此空间复杂度等价于二叉树的高度。

BFS
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        queue = [root]
        depth = 0
        while queue:
            for i in range(len(queue)):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            depth += 1
        return depth
```

* 时间复杂度：$O(n)$，其中 $n$ 为二叉树的节点个数。与方法一同样的分析，每个节点只会被访问一次。
* 空间复杂度：此方法空间的消耗取决于队列存储的元素数量，其在最坏情况下会达到 $O(n)$。

### [Validate Binary Search Tree](https://leetcode.cn/problems/validate-binary-search-tree/)

DFS

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def helper(node, lower=float('-inf'), upper=float('inf')):
            if not node:
                return True

            val = node.val
            if val <= lower or val >= upper:
                return False

            if not helper(node.left, lower, val):
                return False

            if not helper(node.right, val, upper):
                return False

            return True

        return helper(root)
```

* 时间复杂度：$O(n)$，其中 $n$ 为二叉树的节点个数。在递归调用的时候二叉树的每个节点最多被访问一次，因此时间复杂度为 $O(n)$。
* 空间复杂度：$O(n)$，其中 $n$ 为二叉树的节点个数。递归函数在递归过程中需要为每一层递归函数分配栈空间，所以这里需要额外的空间且该空间取决于递归的深度，即二叉树的高度。最坏情况下二叉树为一条链，树的高度为 $n$ ，递归最深达到 $n$ 层，故最坏情况下空间复杂度为 $O(n)$ 。

BFS

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        stack = []
        inorder = float("-inf")

        while stack or root:
            while root:
                stack.append(root)
                root = root.left

            root = stack.pop()
            if root.val <= inorder:
                return False

            inorder = root.val
            root = root.right

        return True
```

* 时间复杂度：$O(n)$，其中 $n$ 为二叉树的节点个数。二叉树的每个节点最多被访问一次，因此时间复杂度为 $O(n)$。
* 空间复杂度：$O(n)$，其中 $n$ 为二叉树的节点个数。栈最多存储 $n$ 个节点，因此需要额外的 $O(n)$ 的空间。

### [Symmetric Tree](https://leetcode.cn/problems/symmetric-tree/)

DFS

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def check(node1, node2):
            if node1 is None and node2 is None:
                return True

            if node1 is None or node2 is None:
                return False

            if node1.val != node2.val:
                return False

            if not check(node1.left, node2.right):
                return False

            if not check(node1.right, node2.left):
                return False

            return True

        return check(root, root)
```

假设树上一共 $n$ 个节点。
* 时间复杂度：这里遍历了这棵树，渐进时间复杂度为 $O(n)$。
* 空间复杂度：这里的空间复杂度和递归使用的栈空间有关，这里递归层数不超过 $n$，故渐进空间复杂度为 $O(n)$。

BFS

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        queue = [(root, root)]
        while queue:
            node1, node2 = queue.pop()
            if node1 is None and node2 is None:
                continue

            if node1 is None or node2 is None:
                return False

            if node1.val != node2.val:
                return False

            queue.append((node1.left, node2.right))
            queue.append((node1.right, node2.left))

        return True
```

* 时间复杂度：这里遍历了这棵树，渐进时间复杂度为 $O(n)$。
* 空间复杂度：这里需要用一个队列来维护节点，每个节点最多进队一次，出队一次，队列中最多不会超过 $n$ 个点，故渐进空间复杂度为 $O(n)$。