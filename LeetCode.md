# [两数之和](https://leetcode.cn/problems/two-sum/)

python
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashtable = dict()
        for i, num in enumerate(nums):
            if target - num in hashtable:
                return [hashtable[target - num], i]
            hashtable[nums[i]] = i
        return []
```

rust
```rust
impl Solution {
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        for (i, num) in nums.iter().enumerate() {
            match map.get(&(target - num)) {
                Some(v) => return vec![*v, i as i32],
                None => map.insert(num, i as i32),
            };
        }
        vec![]
    }
}
```

# [二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

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
use std::cell::RefCell;
use std::rc::Rc;
impl Solution {
    pub fn inorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        enum Visited {
            Yes,
            No,
        }
        let mut res: Vec<i32> = vec![];
        let mut stack = vec![(Visited::No, root)];
        while !stack.is_empty() {
            if let Some(s) = stack.pop() {
                if let Some(node) = s.1.clone() {
                    if let Visited::No = s.0 {
                        stack.push((Visited::No, node.borrow().right.clone()));
                        stack.push((Visited::Yes, s.1.clone()));
                        stack.push((Visited::No, node.borrow().left.clone()));
                    } else {
                        res.push(node.borrow().val)
                    }
                }
            }
        }
        res
    }
}
```

# [对称二叉树](https://leetcode.cn/problems/symmetric-tree/)

python
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# DFS
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if root is None:
            return True

        def dfs(left, right):
            if left is None and right is None:
                return True
            if left is None or right is None:
                return False
            if left.val != right.val:
                return False
            return dfs(left.left, right.right) and dfs(left.right, right.left)

        return dfs(root.left, root.right) 

# BFS
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if root is None:
            return True
        
        queue = [(root.left,root.right)]
        while queue:
            left, right = queue.pop()
            if left is None and right is None:
                continue
            if left is None or right is None:
                return False
            if left.val != right.val:
                return False
            queue.append((left.left,right.right))
            queue.append((left.right,right.left))

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

// DFS
impl Solution {
    pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        fn dfs(
            left: &Option<Rc<RefCell<TreeNode>>>,
            right: &Option<Rc<RefCell<TreeNode>>>,
        ) -> bool {
            match (left, right) {
                (None, None) => true,
                (Some(left), Some(right)) => {
                    if left.borrow().val != right.borrow().val {
                        false
                    } else {
                        dfs(&left.borrow().left, &right.borrow().right)
                            && dfs(&left.borrow().right, &right.borrow().left)
                    }
                }
                _ => false,
            }
        }
        match root {
            None => true,
            Some(root) => dfs(&root.borrow().left, &root.borrow().right),
        }
    }
}

// BFS
impl Solution {
    pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        if let Some(root) = root {
            let mut queue = vec![(root.borrow().left.clone(), root.borrow().right.clone())];
            while let Some((left, right)) = queue.pop() {
                match (left, right) {
                    (None, None) => (),
                    (Some(left), Some(right)) => {
                        if left.borrow().val == right.borrow().val {
                            queue.push((left.borrow().left.clone(), right.borrow().right.clone()));
                            queue.push((left.borrow().right.clone(), right.borrow().left.clone()));
                        } else {
                            return false;
                        }
                    },
                    _ => return false,
                }
            }
        }
        true
    }
}
```

 # [二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

python
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# DFS
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0

        leftHeight = self.maxDepth(root.left)
        rightHeight = self.maxDepth(root.right)

        return max(leftHeight, rightHeight) + 1

# BFS
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0

        queue = [root]
        depth = 0

        while queue:
            n = len(queue)
            for i in range(n):
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

// DFS
impl Solution {
    pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        match root {
            None => 0,
            Some(root) => {
                let leftHeight = Self::max_depth(root.borrow().left.clone());
                let rightHeight = Self::max_depth(root.borrow().right.clone());
                leftHeight.max(rightHeight) + 1
            }
        }
    }
}

// BFS
impl Solution {
    pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        match root {
            None => 0,
            Some(root) => {
                let mut queue = std::collections::VecDeque::new();
                queue.push_back(root);
                let mut depth = 0;
                while !queue.is_empty() {
                    let mut n = queue.len();
                    for i in 0..n {
                        if let Some(node) = queue.pop_front() {
                            if let Some(left) = node.borrow().left.clone() {
                                queue.push_back(left);
                            }
                            if let Some(right) = node.borrow().right.clone() {
                                queue.push_back(right);
                            }
                        }
                    }
                    depth += 1;
                }
                depth
            }
        }
    }
}
```

# [将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)

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
            if left >= right:
                return None

            mid = (left + right) // 2

            root = TreeNode(nums[mid])
            root.left = helper(left, mid)
            root.right = helper(mid + 1, right)
            return root

        return helper(0, len(nums))
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
        fn helper(nums: &Vec<i32>, left: usize, right: usize) -> Option<Rc<RefCell<TreeNode>>> {
            if left >= right {
                return None;
            }
            let mid = (left + right) / 2;
            let mut root = TreeNode::new(nums[mid]);
            root.left = helper(nums, left, mid);
            root.right = helper(nums, mid + 1, right);
            return Some(Rc::new(RefCell::new(root)));
        }
        return helper(&nums, 0, nums.len());
    }
}
```
