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


## Sorting and Searching

## Dynamic Programming