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
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if root is None or (root.left is None and root.right is None):
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
