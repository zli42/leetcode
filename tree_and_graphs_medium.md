### [Binary Tree Inorder Traversal](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

DFS

python
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

```rust
// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
use std::rc::Rc;
use std::cell::RefCell;
impl Solution {
    pub fn inorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let mut res = vec![];
        Self::helper(root, &mut res);
        res
    }

    fn helper(node: Option<Rc<RefCell<TreeNode>>>, res: &mut Vec<i32>) {
        if let Some(node) = node {
            let node = node.borrow();
            Self::helper(node.left.clone(), res);
            res.push(node.val);
            Self::helper(node.right.clone(), res);
        }
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 为二叉树节点的个数。二叉树的遍历中每个节点会被访问一次且只会被访问一次。
* 空间复杂度：$O(n)$。空间复杂度取决于递归的栈深度，而栈深度在二叉树为一条链的情况下会达到 $O(n)$ 的级别。

BFS

python
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

```rust
// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
use std::rc::Rc;
use std::cell::RefCell;
impl Solution {
    pub fn inorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        enum Visited {
            Yes,
            No,
        }

        let mut res = vec![];
        let mut queue = vec![(root, Visited::No)];
        while let Some(each) = queue.pop() {
            if let Some(node) = each.0.clone() {
                let node = node.borrow();
                match each.1 {
                    Visited::Yes => {
                        res.push(node.val);
                    }
                    Visited::No => {
                        queue.push((node.right.clone(), Visited::No));
                        queue.push((each.0.clone(), Visited::Yes));
                        queue.push((node.left.clone(), Visited::No));
                    }
                }
            }
        }
        res
    }
}
```

* 时间复杂度：$O(n)$。
* 空间复杂度：$O(n)$。

### [Binary Tree Zigzag Level Order Traversal](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/)

python
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

rust
```rust
// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
use std::rc::Rc;
use std::cell::RefCell;
impl Solution {
    pub fn zigzag_level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        if let Some(root) = root {
            let mut res = vec![];
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(root);
            let mut level = 1;
            while !queue.is_empty() {
                let mut tmp = vec![];
                for _ in 0..queue.len() {
                    if let Some(node) = queue.pop_front() {
                        let node = node.borrow();
                        tmp.push(node.val);
                        if let Some(left_node) = node.left.clone() {
                            queue.push_back(left_node);
                        }
                        if let Some(right_node) = node.right.clone() {
                            queue.push_back(right_node);
                        }
                    }
                }
                if level % 2 == 0 {
                    tmp.reverse();
                }
                res.push(tmp);
                level += 1;
            }
            res
        } else {
            vec![]
        }
    }
}
```

* 时间复杂度：$O(N)$，其中 $N$ 为二叉树的节点数。每个节点会且仅会被遍历一次。
* 空间复杂度：$O(N)$。我们需要维护存储节点的队列和存储节点值的双端队列，空间复杂度为 $O(N)$。

### [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

python
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

rust
```rust
// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
impl Solution {
    pub fn build_tree(preorder: Vec<i32>, inorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        let in_idx: HashMap<_, _> = inorder.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        Self::helper(
            &preorder,
            &inorder,
            0,
            preorder.len(),
            0,
            inorder.len(),
            &in_idx,
        )
    }

    fn helper(
        preorder: &[i32],
        inorder: &[i32],
        pre_left: usize,
        pre_right: usize,
        in_left: usize,
        in_right: usize,
        in_idx: &HashMap<i32, usize>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if pre_left >= pre_right {
            return None;
        }

        let root_val = preorder[pre_left];
        let mut root = TreeNode::new(root_val);

        let left_size = *in_idx.get(&root_val).unwrap() - in_left;
        root.left = Self::helper(
            &preorder,
            &inorder,
            pre_left + 1,
            pre_left + 1 + left_size,
            in_left,
            in_left + left_size,
            &in_idx,
        );
        root.right = Self::helper(
            &preorder,
            &inorder,
            pre_left + 1 + left_size,
            pre_right,
            in_left + left_size + 1,
            in_right,
            &in_idx,
        );

        Some(Rc::new(RefCell::new(root)))
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 是树中的节点个数。
* 空间复杂度：$O(n)$，除去返回的答案需要的 $O(n)$ 空间之外，我们还需要使用 $O(n)$ 的空间存储哈希映射，以及 $O(h)$（其中 $h$ 是树的高度）的空间表示递归时栈空间。这里 $h < nh<n$，所以总空间复杂度为 $O(n)$。

### [Populating Next Right Pointers in Each Node](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/)

python
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

python
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

python
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

rust
```rust
// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//   pub val: i32,
//   pub left: Option<Rc<RefCell<TreeNode>>>,
//   pub right: Option<Rc<RefCell<TreeNode>>>,
// }
//
// impl TreeNode {
//   #[inline]
//   pub fn new(val: i32) -> Self {
//     TreeNode {
//       val,
//       left: None,
//       right: None
//     }
//   }
// }
use std::rc::Rc;
use std::cell::RefCell;
impl Solution {
    pub fn kth_smallest(root: Option<Rc<RefCell<TreeNode>>>, k: i32) -> i32 {
        enum Visited {
            Yes,
            No,
        }
        let mut k = k;
        let mut queue = vec![(root, Visited::No)];
        while let Some((root, state)) = queue.pop() {
            if let Some(node) = root.clone() {
                let node = node.borrow();
                match state {
                    Visited::Yes => {
                        k -= 1;
                        if k == 0 {
                            return node.val;
                        }
                    }
                    Visited::No => {
                        queue.push((node.right.clone(), Visited::No));
                        queue.push((root, Visited::Yes));
                        queue.push((node.left.clone(), Visited::No));
                    }
                }
            }
        }
        -1
    }
}
```

* 时间复杂度：$O(H+k)$，其中 $H$ 是树的高度。在开始遍历之前，我们需要 $O(H)$ 到达叶结点。当树是平衡树时，时间复杂度取得最小值 $O(\log N + k)$；当树是线性树（树中每个结点都只有一个子结点或没有子结点）时，时间复杂度取得最大值 $O(N+k)$。
* 空间复杂度：$O(H)$，栈中最多需要存储 $H$ 个元素。当树是平衡树时，空间复杂度取得最小值 $O(\log N)$；当树是线性树时，空间复杂度取得最大值 $O(N)$。

### [Number of Islands](https://leetcode.cn/problems/number-of-islands/)

DFS

python
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

rust
```rust
impl Solution {
    pub fn num_islands(grid: Vec<Vec<char>>) -> i32 {
        let mut grid = grid;
        let mut res = 0;
        for row in 0..grid.len() {
            for col in 0..grid[0].len() {
                if grid[row][col] == '1' {
                    Self::helper(&mut grid, row, col);
                    res += 1;
                }
            }
        }
        res
    }

    fn helper(grid: &mut Vec<Vec<char>>, row: usize, col: usize) {
        if row < 0 || row > grid.len() - 1 || col < 0 || col > grid[0].len() - 1 {
            return;
        }
        if grid[row][col] != '1' {
            return;
        }
        grid[row][col] = '2';
        Self::helper(grid, row - 1, col);
        Self::helper(grid, row + 1, col);
        Self::helper(grid, row, col - 1);
        Self::helper(grid, row, col + 1);
    }
}
```

* 时间复杂度：$O(MN)$，其中 $M$ 和 $N$ 分别为行数和列数。
* 空间复杂度：$O(MN)$，在最坏情况下，整个网格均为陆地，深度优先搜索的深度达到 $MN$。

BFS

python
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
