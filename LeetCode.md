- [两数之和](#两数之和)
- [二叉树的中序遍历](#二叉树的中序遍历)
- [对称二叉树](#对称二叉树)
- [二叉树的最大深度](#二叉树的最大深度)
- [将有序数组转换为二叉搜索树](#将有序数组转换为二叉搜索树)
- [两个数组的交集](#两个数组的交集)
- [寻找重复数](#寻找重复数)
- [三数之和](#三数之和)
- [搜索旋转排序数组](#搜索旋转排序数组)
- [旋转图像](#旋转图像)
- [最大子数组和](#最大子数组和)

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
# [两个数组的交集](https://leetcode.cn/problems/intersection-of-two-arrays-ii/)

python
```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if len(nums1) > len(nums2):
            return self.intersect(nums2, nums1)
        
        m = collections.Counter()
        for num in nums1:
            m[num] += 1
        
        intersection = list()
        for num in nums2:
            if (count := m.get(num, 0)) > 0:
                intersection.append(num)
                m[num] -= 1
        
        return intersection
```

rust
```rust
impl Solution {
    pub fn intersect(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
        if nums1.len() > nums2.len() {
            return Self::intersect(nums2, nums1);
        }

        let mut map = std::collections::HashMap::new();
        for num in nums1 {
            let count = map.entry(num).or_insert(0);
            *count += 1;
        }

        let mut intersection = vec![];
        for num in nums2 {
            if let Some(count) = map.get_mut(&num) {
                if *count > 0 {
                    intersection.push(num);
                    *count -= 1;
                }
            }
        }
        intersection
    }
}
```

# [寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/)

python
```python
# 二分
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            # 抽屉原理
            cnt = 0
            for num in nums:
                if num <= mid:
                    cnt += 1
            if cnt <= mid:     
                left = mid+1
            else:
                right = mid
        return left

# 有环链表 快慢指针
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow, fast = 0, 0
        while True:
            slow = nums[slow]           
            fast = nums[nums[fast]]     
            if fast == slow:    
                break
        
        slow = 0                
        while slow != fast:     
            slow = nums[slow]
            fast = nums[fast]
        
        return slow
```

rust
```rust
// 二分
impl Solution {
    pub fn find_duplicate(nums: Vec<i32>) -> i32 {
        let mut left = 0;
        let mut right = nums.len();
        while left < right {
            let mid = (left + right) / 2;
            let mut cnt = 0;
            for num in &nums {
                if *num as usize <= mid {
                    cnt += 1;
                }
            }
            if cnt <= mid {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        left as i32
    }
}

// 有环链表 快慢指针
impl Solution {
    pub fn find_duplicate(nums: Vec<i32>) -> i32 {
        let mut slow = 0;
        let mut fast = 0;
        loop {
            slow = nums[slow] as usize;
            fast = nums[nums[fast] as usize] as usize;
            if fast == slow {
                break;
            }
        }

        slow = 0;
        while slow != fast {
            slow = nums[slow] as usize;
            fast = nums[fast] as usize;
        }
        slow as i32
    }
}
```
# [三数之和](https://leetcode.cn/problems/3sum/)

python
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []
        if not nums or n < 3:
            return res

        nums.sort()
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
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                    r -= 1
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1
                elif s > 0:
                    r -= 1
                else:
                    l += 1
        return res
```

rust
```rust
impl Solution {
    pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let n = nums.len();
        let mut res = vec![];
        if n < 3 {
            return res;
        }
        
        let mut nums = nums;
        nums.sort();
        
        for i in 0..n {
            if nums[i] > 0 {
                return res;
            }
            
            if i > 0 && nums[i] == nums[i - 1] {
                continue;
            }

            let mut l = i + 1;
            let mut r = n - 1;
            while l < r {
                let s = nums[i] + nums[l] + nums[r];
                if s == 0 {
                    res.push(vec![nums[i], nums[l], nums[r]]);
                    l += 1;
                    while l < r && nums[l] == nums[l - 1] {
                        l += 1;
                    }
                    r -= 1;
                    while l < r && nums[r] == nums[r + 1] {
                        r -= 1;
                    }
                } else if s > 0 {
                    r -= 1;
                } else {
                    l += 1;
                }
            }
        }
        return res;
    }
}
```

# [搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

python
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
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
                if nums[mid] < target <= nums[len(nums) - 1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1
```

rust
```rust
impl Solution {
    pub fn search(nums: Vec<i32>, target: i32) -> i32 {
        let n = nums.len() - 1;
        let mut l = 0;
        let mut r = n;
        while l <= r {
            let mid = (l + r) / 2;
            if nums[mid] == target {
                return mid as i32;
            }
            if nums[0] <= nums[mid] {
                if nums[0] <= target && target <= nums[mid] {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if nums[mid] < target && target <= nums[n] {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return -1;
    }
}
```

# [旋转图像](https://leetcode.cn/problems/rotate-image/)

python
```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
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

rust
```rust
impl Solution {
    pub fn rotate(matrix: &mut Vec<Vec<i32>>) {
        let n = matrix.len();
        for i in 0..n / 2 {
            for j in 0..n {
                let tmp = matrix[i][j];
                matrix[i][j] = matrix[n - i - 1][j];
                matrix[n - i - 1][j] = tmp;
            }
        }
        for i in 0..n {
            for j in 0..i {
                let tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }
    }
}
```

# [最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

python
```python
from typing import List


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        size = len(nums)
        pre = 0
        res = nums[0]
        for i in range(size):
            pre = max(nums[i], pre + nums[i])
            res = max(res, pre)
        return res
```

rust
```rust
impl Solution {
    pub fn max_sub_array(nums: Vec<i32>) -> i32 {
        let mut pre = 0;
        let mut res = nums[0];
        for num in nums {
            pre = std::cmp::max(pre + num, num);
            if num > 0 
            res = std::cmp::max(res, pre);
        }
        res
    }
}
```