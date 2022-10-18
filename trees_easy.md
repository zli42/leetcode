### [Maximum Depth of Binary Tree](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

DFS

python
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
    pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        if let Some(root) = root {
            let root = root.borrow();
            let left_depth = Self::max_depth(root.left.clone());
            let right_depth = Self::max_depth(root.right.clone());
            std::cmp::max(left_depth, right_depth) + 1
        } else {
            0
        }
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 为二叉树节点的个数。每个节点在递归中只被遍历一次。
* 空间复杂度：$O(height)$，其中 `height` 表示二叉树的高度。递归函数需要栈空间，而栈空间取决于递归的深度，因此空间复杂度等价于二叉树的高度。

BFS
python
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
    pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        if let Some(root) = root {
            let mut depth = 0;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(root);
            while !queue.is_empty() {
                for _ in 0..queue.len() {
                    if let Some(node) = queue.pop_front() {
                        let node = node.borrow();
                        if let Some(left_node) = node.left.clone() {
                            queue.push_back(left_node);
                        }
                        if let Some(right_node) = node.right.clone() {
                            queue.push_back(right_node);
                        }
                    }
                }
                depth += 1;
            }
            depth
        } else {
            0
        }
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 为二叉树的节点个数。与方法一同样的分析，每个节点只会被访问一次。
* 空间复杂度：此方法空间的消耗取决于队列存储的元素数量，其在最坏情况下会达到 $O(n)$。

### [Validate Binary Search Tree](https://leetcode.cn/problems/validate-binary-search-tree/)

DFS

python
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

            return helper(node.left, lower, val) and helper(node.right, val, upper)

        return helper(root)
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
    pub fn is_valid_bst(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        Self::helper(&root, std::i32::MIN, std::i32::MAX)
    }

    fn helper(node: &Option<Rc<RefCell<TreeNode>>>, left: i32, right: i32) -> bool {
        if let Some(node) = node {
            let node = node.borrow();
            let val = node.val;
            if val <= left || val >= right {
                return false;
            }
            Self::helper(&node.left, left, val) && Self::helper(&node.right, val, right)
        } else {
            true
        }
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 为二叉树的节点个数。在递归调用的时候二叉树的每个节点最多被访问一次，因此时间复杂度为 $O(n)$。
* 空间复杂度：$O(n)$，其中 $n$ 为二叉树的节点个数。递归函数在递归过程中需要为每一层递归函数分配栈空间，所以这里需要额外的空间且该空间取决于递归的深度，即二叉树的高度。最坏情况下二叉树为一条链，树的高度为 $n$ ，递归最深达到 $n$ 层，故最坏情况下空间复杂度为 $O(n)$ 。

BFS

python
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
    pub fn is_valid_bst(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        enum Visited {
            Yes,
            No,
        }
        let mut pre = std::i32::MIN;
        let mut queue = vec![(root, Visited::No)];
        while let Some(item) = queue.pop() {
            if let Some(node) = item.0.clone() {
                let node = node.borrow();
                match item.1 {
                    Visited::Yes => {
                        let val = node.val;
                        if val <= pre {
                            return false;
                        }
                        pre = val;
                    }
                    Visited::No => {
                        queue.push((node.right.clone(), Visited::No));
                        queue.push((item.0.clone(), Visited::Yes));
                        queue.push((node.left.clone(), Visited::No));
                    }
                }
            }
        }
        true
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 为二叉树的节点个数。二叉树的每个节点最多被访问一次，因此时间复杂度为 $O(n)$。
* 空间复杂度：$O(n)$，其中 $n$ 为二叉树的节点个数。栈最多存储 $n$ 个节点，因此需要额外的 $O(n)$ 的空间。

### [Symmetric Tree](https://leetcode.cn/problems/symmetric-tree/)

DFS

python
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

            return check(node1.left, node2.right) and check(node1.right, node2.left)

        return check(root, root)
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
    pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        if let Some(root) = root {
            let root = root.borrow();
            Self::helper(&root.left, &root.right)
        } else {
            true
        }
    }

    fn helper(
        left_node: &Option<Rc<RefCell<TreeNode>>>,
        right_node: &Option<Rc<RefCell<TreeNode>>>,
    ) -> bool {
        match (left_node, right_node) {
            (Some(left_node), Some(right_node)) => {
                let left_node = left_node.borrow();
                let right_node = right_node.borrow();
                if left_node.val != right_node.val {
                    return false;
                }
                Self::helper(&left_node.left, &right_node.right)
                    && Self::helper(&left_node.right, &right_node.left)
            }
            (None, None) => true,
            _ => false,
        }
    }
}
```

假设树上一共 $n$ 个节点。
* 时间复杂度：这里遍历了这棵树，渐进时间复杂度为 $O(n)$。
* 空间复杂度：这里的空间复杂度和递归使用的栈空间有关，这里递归层数不超过 $n$，故渐进空间复杂度为 $O(n)$。

BFS

python
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
    pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        if let Some(root) = root {
            let root = root.borrow();
            let mut queue = vec![(root.left.clone(), root.right.clone())];
            while let Some((left_node, right_node)) = queue.pop() {
                match (left_node, right_node) {
                    (Some(left_node), Some(right_node)) => {
                        let left_node = left_node.borrow();
                        let right_node = right_node.borrow();
                        if left_node.val != right_node.val {
                            return false;
                        }
                        queue.push((left_node.left.clone(), right_node.right.clone()));
                        queue.push((left_node.right.clone(), right_node.left.clone()));
                    }
                    (None, None) => (),
                    _ => return false,
                }
            }
        }
        true
    }
}
```

* 时间复杂度：这里遍历了这棵树，渐进时间复杂度为 $O(n)$。
* 空间复杂度：这里需要用一个队列来维护节点，每个节点最多进队一次，出队一次，队列中最多不会超过 $n$ 个点，故渐进空间复杂度为 $O(n)$。

### [Binary Tree Level Order Traversal](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

python
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = []
        if not root:
            return res

        queue = [root]
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
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
    pub fn level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        if let Some(root) = root {
            let mut res = vec![];
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(root);
            while !queue.is_empty() {
                let mut level = vec![];
                for _ in 0..queue.len() {
                    if let Some(node) = queue.pop_front() {
                        let node = node.borrow();
                        level.push(node.val);
                        if let Some(left_node) = node.left.clone() {
                            queue.push_back(left_node);
                        }
                        if let Some(right_node) = node.right.clone() {
                            queue.push_back(right_node);
                        }
                    }
                }
                res.push(level);
            }
            res
        } else {
            vec![]
        }
    }
}
```

* 时间复杂度：每个点进队出队各一次，故渐进时间复杂度为 $O(n)$。
* 空间复杂度：队列中元素的个数不超过 $n$ 个，故渐进空间复杂度为 $O(n)$。

### [Convert Sorted Array to Binary Search Tree](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)

python
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def helper(left, right):
            if left > right:
                return None
            
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            root.left = helper(left, mid - 1)
            root.right = helper(mid + 1, right)
            return root
        return helper(0, len(nums) - 1)     
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
    pub fn sorted_array_to_bst(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        Self::helper(&nums, 0, nums.len())
    }

    fn helper(nums: &Vec<i32>, left: usize, right: usize) -> Option<Rc<RefCell<TreeNode>>> {
        if left >= right {
            return None;
        }

        let mid = left + (right - left) / 2;
        let mut root = TreeNode::new(nums[mid]);
        root.left = Self::helper(&nums, left, mid);
        root.right = Self::helper(&nums, mid + 1, right);
        return Some(Rc::new(RefCell::new(root)));
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 是数组的长度。每个数字只访问一次。
* 空间复杂度：$O(\log{n})$，其中 $n$ 是数组的长度。空间复杂度不考虑返回值，因此空间复杂度主要取决于递归栈的深度，递归栈的深度是 $O(\log{n})$。
