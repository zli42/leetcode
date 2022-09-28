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

### [Increasing Triplet Subsequence](https://leetcode.cn/problems/increasing-triplet-subsequence/)

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        n = len(nums)
        if n < 3:
            return False
        
        first = nums[0]
        second = float('inf')
        for i in range(1, n):
            third = nums[i]
            if third > second:
                return True
            
            if third > first:
                second = third
            else:
                first = third
        return False
```

* 时间复杂度：$O(n)$，其中 $n$ 是数组 `nums` 的长度。需要遍历数组一次。
* 空间复杂度：$O(1)$。

### [Count and Say](https://leetcode.cn/problems/count-and-say/)

```python
class Solution:
    def countAndSay(self, n: int) -> str:
        res = [1]
        for _ in range(1, n):
            cur = []
            pos = 0
            m = len(res)
            while pos < m:
                cnt = 0
                dig = res[pos]
                while pos < m and res[pos] == dig:
                    cnt += 1
                    pos += 1
                cur.append(cnt)
                cur.append(dig)
            res = cur
        return ''.join(str(e) for e in res)
```

* 时间复杂度：$O(N \times M)$，其中 $N$ 为给定的正整数，$M$ 为生成的字符串中的最大长度。
* 空间复杂度：O$(M)$。其中 $M$ 为生成的字符串中的最大长度。

## Tree and Graphs

### [Binary Tree Inorder Traversal](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

DFS

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
	def inorderTraversal(self, root):
		res = []
		def dfs(node):
			if not node:
				return
			dfs(node.left)
			res.append(node.val)
			dfs(node.right)
		dfs(root)
		return res
```

* 时间复杂度：$O(n)$，其中 $n$ 为二叉树节点的个数。二叉树的遍历中每个节点会被访问一次且只会被访问一次。
* 空间复杂度：$O(n)$。空间复杂度取决于递归的栈深度，而栈深度在二叉树为一条链的情况下会达到 $O(n)$ 的级别。

BFS

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        NEW_NODE, VISITED_NODE = False, True
        res = []
        stack = [(NEW_NODE, root)]
        while stack:
            state, node = stack.pop()
            if node is None: continue
            if state == NEW_NODE:
                stack.append((NEW_NODE, node.right))
                stack.append((VISITED_NODE, node))
                stack.append((NEW_NODE, node.left))
            else:
                res.append(node.val)
        return res
```

* 时间复杂度：$O(n)$。
* 空间复杂度：$O(n)$。

### [Binary Tree Zigzag Level Order Traversal](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        res = []
        level = 0
        queue = [root]
        while queue:
            tmp = collections.deque()
            for _ in range(len(queue)):
                node = queue.pop(0)
                if level % 2 == 0:
                    tmp.append(node.val)
                else:
                    tmp.appendleft(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(tmp)
            level += 1
        return res
```

* 时间复杂度：$O(N)$，其中 $N$ 为二叉树的节点数。每个节点会且仅会被遍历一次。
* 空间复杂度：$O(N)$。我们需要维护存储节点的队列和存储节点值的双端队列，空间复杂度为 $O(N)$。

### [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        
        root = TreeNode(preorder[0])
        mid = inorder.index(root.val)
        root.left = self.buildTree(preorder[1:mid+1], inorder[:mid])
        root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])
        return root
```

* 时间复杂度：$O(n)$，其中 $n$ 是树中的节点个数。
* 空间复杂度：$O(n)$，除去返回的答案需要的 $O(n)$ 空间之外，我们还需要使用 $O(n)$ 的空间存储哈希映射，以及 $O(h)$（其中 $h$ 是树的高度）的空间表示递归时栈空间。这里 $h < nh<n$，所以总空间复杂度为 $O(n)$。

### [Populating Next Right Pointers in Each Node](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return root
        
        queue = collections.deque([root])
        while queue:
            s = len(queue)
            for i in range(s):
                node = queue.popleft()
                if i < s - 1:
                    node.next = queue[0]
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return root
```

* 时间复杂度：$O(N)$。每个节点会被访问一次且只会被访问一次，即从队列中弹出，并建立 `next` 指针。
* 空间复杂度：$O(N)$。这是一棵完美二叉树，它的最后一个层级包含 $N/2$ 个节点。广度优先遍历的复杂度取决于一个层级上的最大元素数量。这种情况下空间复杂度为 $O(N)$。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return root
        
        leftmost = root
        while leftmost.left:
            head = leftmost
            while head:
                head.left.next = head.right
                if head.next:
                    head.right.next = head.next.left
                head = head.next
            leftmost = leftmost.left
        return root
```

* 时间复杂度：$O(N)$，每个节点只访问一次。
* 空间复杂度：$O(1)$，不需要存储额外的节点。

### [Kth Smallest Element in a BST](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        res = []
        NEW = True
        VISITED = False
        stack = [(NEW, root)]
        while stack:
            state, node = stack.pop()
            if not node:
                continue
            if state == NEW:
                stack.append((NEW, node.right))
                stack.append((VISITED, node))
                stack.append((NEW, node.left))
            else:
                res.append(node.val)
                if len(res) == k:
                    return node.val
        return None
```

* 时间复杂度：$O(H+k)$，其中 $H$ 是树的高度。在开始遍历之前，我们需要 $O(H)$ 到达叶结点。当树是平衡树时，时间复杂度取得最小值 $O(\log N + k)$；当树是线性树（树中每个结点都只有一个子结点或没有子结点）时，时间复杂度取得最大值 $O(N+k)$。
* 空间复杂度：$O(H)$，栈中最多需要存储 $H$ 个元素。当树是平衡树时，空间复杂度取得最小值 $O(\log N)$；当树是线性树时，空间复杂度取得最大值 $O(N)$。

### [Number of Islands](https://leetcode.cn/problems/number-of-islands/)

DFS

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(grid, r, c):
            grid[r][c] = '2'
            for x, y in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]:
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == '1':
                    dfs(grid, x, y)
                    
        num_islands = 0
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == '1':
                    num_islands += 1
                    dfs(grid, r, cO)
                    
        return num_islands
```

* 时间复杂度：$O(MN)$，其中 $M$ 和 $N$ 分别为行数和列数。
* 空间复杂度：$O(MN)$，在最坏情况下，整个网格均为陆地，深度优先搜索的深度达到 $MN$。

BFS

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        num_islands = 0
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == '1':
                    num_islands += 1
                    grid[r][c] = '2'
                    neighbors = collections.deque([(r, c)])
                    while neighbors:
                        row, col = neighbors.popleft()
                        for x, y in [(row+1, col), (row-1, col), (row, col+1), (row, col-1)]:
                            if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == '1':
                                neighbors.append((x, y))
                                grid[x][y] = '2'
                    
        return num_islands
```

* 时间复杂度：$O(MN)$，其中 $M$ 和 $N$ 分别为行数和列数。
* 空间复杂度：$O(\min(M, N))$，在最坏情况下，整个网格均为陆地，队列的大小可以达到 $\min(M, N)$。

## Sorting and Searching

### [Sort Colors](https://leetcode.cn/problems/sort-colors/)

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        p0 = 0
        p1 = 0
        for i in range(len(nums)):
            if nums[i] == 1:
                nums[i], nums[p1] = nums[p1], nums[i]
                p1 += 1
            elif nums[i] ==0:
                nums[i], nums[p0] = nums[p0], nums[i]
                if p0 < p1:
                    nums[i], nums[p1] = nums[p1], nums[i]
                p0 += 1
                p1 += 1
```

* 时间复杂度：$O(n)$，其中 $n$ 是数组 `nums` 的长度。
* 空间复杂度：$O(1)$。

### [Top K Frequent Elements](https://leetcode.cn/problems/top-k-frequent-elements/)

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counter = {}
        for num in nums:
            counter[num] = counter.get(num, 0) + 1

        import heapq

        heap = []
        for num, cnt in counter.items():
            if len(heap) < k:
                heapq.heappush(heap, (cnt, num))
            elif cnt > heap[0][0]:
                heapq.heapreplace(heap, (cnt, num))

        return [each[1] for each in heap]
```

* 时间复杂度：$O(N\log k)$，其中 $N$ 为数组的长度。我们首先遍历原数组，并使用哈希表记录出现次数，每个元素需要 $O(1)$ 的时间，共需 $O(N)$ 的时间。随后，我们遍历「出现次数数组」，由于堆的大小至多为 $k$，因此每次堆操作需要 $O(\log k)$ 的时间，共需 $O(N\log k)$ 的时间。二者之和为 $O(N\log k)$。
* 空间复杂度：$O(N)$。哈希表的大小为 $O(N)$，而堆的大小为 $O(k)$，共计为 $O(N)$。

### [Kth Largest Element in an Array](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        import random

        def quickFind(nums, k, left, right):
            rand = random.randint(left, right)
            nums[right], nums[rand] = nums[rand], nums[right]
            pivot = nums[right]

            i = left
            for j in range(left, right):
                if nums[j] <= pivot:
                    nums[j], nums[i] = nums[i], nums[j]
                    i += 1
            nums[right], nums[i] = nums[i], nums[right]

            if i == k:
                return nums[i]
            elif i < k:
                return quickFind(nums, k, i + 1, right)
            elif i > k:
                return quickFind(nums, k, left, i - 1)

        n = len(nums)
        return quickFind(nums, n - k, 0, n - 1)
```

* 时间复杂度：$O(n)$。使用快速排序，平均时间复杂度是 $O(n \log n)$，最坏的时间代价是 $O(n ^ 2)$。把原来递归两个区间变成只递归一个区间，提高了时间效率。这就是「快速选择」算法。
* 空间复杂度：$O(\log n)$，递归使用栈空间的空间代价的期望为 $O(\log n)$。

### [Find Peak Element](https://leetcode.cn/problems/find-peak-element/)

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        def get(nums, i):
            if i == -1 or i == len(nums):
                return float('-inf')
            return nums[i]
        
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if get(nums, mid-1) < get(nums, mid) > get(nums, mid+1):
                return mid
            if get(nums, mid) < get(nums, mid-1):
                right = mid - 1
            else:
                left = mid + 1
```

* 时间复杂度：$O(\log n)$，其中 $n$ 是数组 `nums` 的长度。
* 空间复杂度：$O(1)$。

### [Merge Intervals](https://leetcode.cn/problems/merge-intervals/)

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged
```

* 时间复杂度：$O(n\log n)$，其中 $n$ 为区间的数量。除去排序的开销，我们只需要一次线性扫描，所以主要的时间开销是排序的 $O(n\log n)$。
* 空间复杂度：$O(\log n)$，其中 $n$ 为区间的数量。这里计算的是存储答案之外，使用的额外空间。$O(\log n)$ 即为排序所需要的空间复杂度。

### [Search in Rotated Sorted Array](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            
            if nums[0] <= nums[mid]:
                if nums[0] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
                    
        return -1
```

* 时间复杂度： $O(\log n)$，其中 $n$ 为 `nums` 数组的大小。整个算法时间复杂度即为二分查找的时间复杂度 $O(\log n)$。
* 空间复杂度： $O(1)$ 。我们只需要常数级别的空间存放变量。
        
### [Search a 2D Matrix II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        x = 0
        y = len(matrix[0]) - 1
        while x < len(matrix) and y >= 0:
            if matrix[x][y] == target:
                return True
            
            if matrix[x][y] < target:
                x += 1
            else:
                y -= 1
                
        return False
```

* 时间复杂度：$O(m+n)$。在搜索的过程中，如果我们没有找到 `target`，那么我们要么将 `y` 减少 `1`，要么将 `x` 增加 `1`。由于 `(x, y)` 的初始值分别为 `(0, n-1)`，因此 `y` 最多能被减少 `n` 次，`x` 最多能被增加 `m` 次，总搜索次数为 `m + n`。在这之后，`x` 和 `y` 就会超出矩阵的边界。
* 空间复杂度：`O(1)`。

## Dynamic Programming