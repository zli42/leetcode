- [两数之和](#两数之和)
  - [哈希表](#哈希表)
    - [python](#python)
    - [rust](#rust)
- [删除有序数组中的重复项](#删除有序数组中的重复项)
  - [双指针](#双指针)
    - [python](#python-1)
    - [rust](#rust-1)
- [合并两个有序数组](#合并两个有序数组)
  - [双指针](#双指针-1)
    - [python](#python-2)
    - [rust](#rust-2)
- [二叉树的中序遍历](#二叉树的中序遍历)
  - [递归](#递归)
    - [python](#python-3)
    - [rust](#rust-3)
  - [带状态的迭代](#带状态的迭代)
    - [python](#python-4)
    - [rust](#rust-4)
- [对称二叉树](#对称二叉树)
  - [递归](#递归-1)
    - [python](#python-5)
    - [rust](#rust-5)
  - [迭代](#迭代)
    - [python](#python-6)
    - [rust](#rust-6)
- [二叉树的最大深度](#二叉树的最大深度)
  - [递归](#递归-2)
    - [python](#python-7)
    - [rust](#rust-7)
  - [迭代](#迭代-1)
    - [python](#python-8)
    - [rust](#rust-8)
- [将有序数组转换为二叉搜索树](#将有序数组转换为二叉搜索树)
  - [二叉搜索树](#二叉搜索树)
    - [python](#python-9)
    - [rust](#rust-9)
- [买卖股票的最佳时机](#买卖股票的最佳时机)
  - [动态规划](#动态规划)
    - [python](#python-10)
    - [rust](#rust-10)
- [两个数组的交集](#两个数组的交集)
  - [哈希表](#哈希表-1)
    - [python](#python-11)
    - [rust](#rust-11)
- [寻找重复数](#寻找重复数)
- [盛最多水的容器](#盛最多水的容器)
  - [双指针](#双指针-2)
    - [python](#python-12)
    - [rust](#rust-12)
- [三数之和](#三数之和)
  - [双指针](#双指针-3)
    - [python](#python-13)
    - [rust](#rust-13)
- [搜索旋转排序数组](#搜索旋转排序数组)
  - [二分查找](#二分查找)
    - [python](#python-14)
    - [rust](#rust-14)
- [全排列](#全排列)
  - [回溯](#回溯)
    - [python](#python-15)
    - [rust](#rust-15)
- [最大子数组和](#最大子数组和)
  - [动态规划](#动态规划-1)
    - [python](#python-16)
    - [rust](#rust-16)
- [跳跃游戏](#跳跃游戏)
  - [贪心](#贪心)
    - [python](#python-17)
    - [rust](#rust-17)
- [合并区间](#合并区间)
  - [贪心](#贪心-1)
    - [python](#python-18)
    - [rust](#rust-18)

# [两数之和](https://leetcode.cn/problems/two-sum/)

## 哈希表

### python

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

### rust

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

* 时间复杂度：$O(N)$，其中 $N$ 是数组中的元素数量。对于每一个元素 `x`，我们可以 $O(1)$ 地寻找 `target - x`。
* 空间复杂度：$O(N)$，其中 $N$ 是数组中的元素数量。主要为哈希表的开销。

# [删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)

## 双指针

### python

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        fast = slow = 1
        while fast < n:
            if nums[fast] != nums[fast - 1]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        
        return slow
```

### rust

```rust
impl Solution {
    pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
        let n = nums.len();
        let mut fast = 1;
        let mut slow = 1;
        while fast < n {
            if nums[fast] != nums[fast - 1] {
                nums[slow] = nums[fast];
                slow += 1;
            }
            fast += 1;
        }
        slow as i32
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 是数组的长度。快指针和慢指针最多各移动 $n$ 次。
* 空间复杂度：$O(1)$。只需要使用常数的额外空间。

# [合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/)

## 双指针

### python

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1, p2 = m - 1, n - 1
        tail = m + n - 1
        while p1 >= 0 or p2 >= 0:
            if p1 == -1:
                nums1[tail] = nums2[p2]
                p2 -= 1
            elif p2 == -1:
                nums1[tail] = nums1[p1]
                p1 -= 1
            elif nums1[p1] > nums2[p2]:
                nums1[tail] = nums1[p1]
                p1 -= 1
            else:
                nums1[tail] = nums2[p2]
                p2 -= 1
            tail -= 1
```

### rust

```rust
impl Solution {
    pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
        let mut p1 = m as usize;
        let mut p2 = n as usize;
        let mut tail = (m + n - 1) as usize;
        while p1 > 0 || p2 > 0 {
            if p1 == 0 {
                nums1[tail] = nums2[p2 - 1];
                p2 -= 1;
            } else if p2 == 0 {
                nums1[tail] = nums1[p1 - 1];
                p1 -= 1;
            } else if nums1[p1 - 1] > nums2[p2 - 1] {
                nums1[tail] = nums1[p1 - 1];
                p1 -= 1;
            } else {
                nums1[tail] = nums2[p2 - 1];
                p2 -= 1;
            }
            tail -= 1
        }
    }
}
```

* 时间复杂度：$O(m+n)$。指针移动单调递减，最多移动 $m+n$ 次，因此时间复杂度为 $O(m+n)$。
* 空间复杂度：$O(1)$。直接对数组 $nums_1$ 原地修改，不需要额外空间。

# [二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

## 递归

### python

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

### rust

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
        fn dfs(node: Option<Rc<RefCell<TreeNode>>>, res: &mut Vec<i32>) {
            if let Some(node) = node {
                dfs(node.borrow().left.clone(), res);
                res.push(node.borrow().val);
                dfs(node.borrow().right.clone(), res);
            }
        }
        let mut res = vec![];
        dfs(root, &mut res);
        res
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 为二叉树节点的个数。二叉树的遍历中每个节点会被访问一次且只会被访问一次。
* 空间复杂度：$O(n)$。空间复杂度取决于递归的栈深度，而栈深度在二叉树为一条链的情况下会达到 $O(n)$ 的级别。


## 带状态的迭代

### python

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

### rust

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

* 时间复杂度：$O(n)$。
* 空间复杂度：$O(n)$。

# [对称二叉树](https://leetcode.cn/problems/symmetric-tree/)

## 递归

### python

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
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
```

### rust

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
```

* 时间复杂度：这里遍历了这棵树，渐进时间复杂度为 $O(n)$。
* 空间复杂度：这里的空间复杂度和递归使用的栈空间有关，这里递归层数不超过 $n$，故渐进空间复杂度为 $O(n)$。

## 迭代 

### python

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
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

### rust

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

* 时间复杂度：这里遍历了这棵树，渐进时间复杂度为 $O(n)$。
* 空间复杂度：这里需要用一个队列来维护节点，每个节点最多进队一次，出队一次，队列中最多不会超过 $n$ 个点，故渐进空间复杂度为 $O(n)$。

# [二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

## 递归

### python

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0

        leftHeight = self.maxDepth(root.left)
        rightHeight = self.maxDepth(root.right)

        return max(leftHeight, rightHeight) + 1
```

### rust

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
```

* 时间复杂度：$O(n)$，其中 $n$ 为二叉树节点的个数。每个节点在递归中只被遍历一次。
* 空间复杂度：$O(height)$，其中 $height$ 表示二叉树的高度。递归函数需要栈空间，而栈空间取决于递归的深度，因此空间复杂度等价于二叉树的高度。

## 迭代

### python

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
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

### rust

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
* 时间复杂度：$O(n)$，其中 $n$ 为二叉树的节点个数。与方法一同样的分析，每个节点只会被访问一次。
* 空间复杂度：此方法空间的消耗取决于队列存储的元素数量，其在最坏情况下会达到 $O(n)$。

# [将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)

## 二叉搜索树

### python

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

### rust

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

* 时间复杂度：$O(n)$，其中 $n$ 是数组的长度。每个数字只访问一次。
* 空间复杂度：$O(\log{n})$，其中 $n$ 是数组的长度。空间复杂度不考虑返回值，因此空间复杂度主要取决于递归栈的深度，递归栈的深度是 $O(\log{n})$。

# [买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

## 动态规划

### python

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        maxprofit = 0
        minprice = prices[0]
        for price in prices:
            maxprofit = max(price - minprice, maxprofit)
            minprice = min(price, minprice)
        return maxprofit
```

### rust

```rust
impl Solution {
    pub fn max_profit(prices: Vec<i32>) -> i32 {
        let mut maxprofit = 0;
        let mut minprice = prices[0];
        for price in prices {
            maxprofit = std::cmp::max(price - minprice, maxprofit);
            minprice = std::cmp::min(price, minprice);
        }
        maxprofit
    }
}
```

* 时间复杂度：$O(n)$，只需要遍历一次。
* 空间复杂度：$O(1)$，只使用了常数个变量。

# [两个数组的交集](https://leetcode.cn/problems/intersection-of-two-arrays-ii/)

## 哈希表

### python

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

### rust

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

* 时间复杂度：$O(m+n)$，其中 $m$ 和 $n$ 分别是两个数组的长度。需要遍历两个数组并对哈希表进行操作，哈希表操作的时间复杂度是 $O(1)$，因此总时间复杂度与两个数组的长度和呈线性关系。
* 空间复杂度：$O(min(m,n))$，其中 $m$ 和 $n$ 分别是两个数组的长度。对较短的数组进行哈希表的操作，哈希表的大小不会超过较短的数组的长度。为返回值创建一个数组 `intersection`，其长度为较短的数组的长度。

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

# [盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

## 双指针

### python

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        ans = 0
        while l < r:
            area = min(height[l], height[r]) * (r - l)
            ans = max(ans, area)
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return ans
```

### rust

```rust
impl Solution {
    pub fn max_area(height: Vec<i32>) -> i32 {
        let mut l = 0;
        let mut r = height.len() - 1;
        let mut ans = 0;
        while l < r {
            let area = std::cmp::min(height[l], height[r]) * (r - l) as i32;
            ans = std::cmp::max(ans, area);
            if height[l] <= height[r] {
                l += 1;
            } else {
                r -= 1;
            }
        }
        ans
    }
}
```

* 时间复杂度：$O(N)$，双指针总计最多遍历整个数组一次。
* 空间复杂度：$O(1)$，只需要额外的常数级别的空间。

# [三数之和](https://leetcode.cn/problems/3sum/)

## 双指针

### python

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

### rust

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

* 时间复杂度：$O(N^2)$，其中 $N$ 是数组 $nums$ 的长度。
* 空间复杂度：$O(logN)$。我们忽略存储答案的空间，额外的排序的空间复杂度为 $O(logN)$。然而我们修改了输入的数组 $nums$，在实际情况下不一定允许，因此也可以看成使用了一个额外的数组存储了 $nums$ 的副本并进行排序，空间复杂度为 $O(N)$。

# [搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

## 二分查找

### python

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

### rust

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

* 时间复杂度： $O(\log{n})$，其中 $n$ 为 $nums$ 数组的大小。整个算法时间复杂度即为二分查找的时间复杂度 $O(\log{n})$。
* 空间复杂度： $O(1)$ 。我们只需要常数级别的空间存放变量。

# [全排列](https://leetcode.cn/problems/permutations/)

## 回溯

### python

```python
class Solution:
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def backtrack(first = 0):
            # 所有数都填完了
            if first == n:  
                res.append(nums[:])
            for i in range(first, n):
                # 动态维护数组
                nums[first], nums[i] = nums[i], nums[first]
                # 继续递归填下一个数
                backtrack(first + 1)
                # 撤销操作
                nums[first], nums[i] = nums[i], nums[first]
        
        n = len(nums)
        res = []
        backtrack()
        return res
```

### rust

```rust
impl Solution {
    pub fn permute(nums: Vec<i32>) -> Vec<Vec<i32>> {
        fn backtrack(first: usize, nums: &mut Vec<i32>, res: &mut Vec<Vec<i32>>) {
            let n = nums.len();
            if first == n {
                res.push(nums.to_vec());
            } else {
                for i in first..n {
                    nums.swap(i, first);
                    backtrack(first + 1, nums, res);
                    nums.swap(i, first);
                }
            }
        }
        let mut nums = nums;
        let mut res = Vec::new();
        backtrack(0, &mut nums, &mut res);
        res
    }
}
```

* 时间复杂度：$O(n \times n!)$，其中 $n$ 为序列的长度。算法的复杂度首先受 `backtrack` 的调用次数制约，`backtrack` 的调用次数为 $\sum_{k = 1}^{n}{P(n, k)}$ 次，其中 $P(n, k) = \frac{n!}{(n - k)!} = n (n - 1) \dots (n - k + 1)$ ，该式被称作 n 的 k - 排列，或者部分排列。而 $\sum_{k = 1}^{n}{P(n, k)} = n! + \frac{n!}{1!} + \frac{n!}{2!} + \frac{n!}{3!} + \ldots + \frac{n!}{(n-1)!} < 2n! + \frac{n!}{2} + \frac{n!}{2^2} + \frac{n!}{2^{n-2}} < 3n!$ 这说明 `backtrack` 的调用次数是 $O(n!)$ 的。而对于 `backtrack` 调用的每个叶结点（共 $n!$ 个），我们需要将当前答案使用 $O(n)$ 的时间复制到答案数组中，相乘得时间复杂度为 $O(n \times n!)$。因此时间复杂度为 $O(n \times n!)$。
* 空间复杂度：$O(n)$，其中 $n$ 为序列的长度。除答案数组以外，递归函数在递归过程中需要为每一层递归函数分配栈空间，所以这里需要额外的空间且该空间取决于递归的深度，这里可知递归调用深度为 $O(n)$。

# [最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

## 动态规划

### python

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        pre = 0
        res = nums[0]
        for num in nums:
            pre = max(num, pre + num)
            res = max(res, pre)
        return res
```

### rust

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

* 时间复杂度：$O(n)$，其中 $n$ 为 `nums` 数组的长度。我们只需要遍历一遍数组即可求得答案。
* 空间复杂度：$O(1)$。我们只需要常数空间存放若干变量。


# [跳跃游戏](https://leetcode.cn/problems/jump-game/)

## 贪心

### python

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n, rightmost = len(nums), 0
        for i in range(n):
            if i <= rightmost:
                rightmost = max(rightmost, i + nums[i])
                if rightmost >= n - 1:
                    return True
        return False
```

### rust

```rust
impl Solution {
    pub fn can_jump(nums: Vec<i32>) -> bool {
        let n = nums.len();
        let mut rightmost = 0;
        for i in 0..n {
            if i <= rightmost {
                rightmost = std::cmp::max(rightmost, i + nums[i] as usize);
                if rightmost >= n - 1 {
                    return true;
                }
            }
        }
        return false;
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 为数组的大小。只需要访问 nums 数组一遍，共 $n$ 个位置。
* 空间复杂度：$O(1)$，不需要额外的空间开销。

# [合并区间](https://leetcode.cn/problems/merge-intervals/)

## 贪心

### python

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # 如果列表为空，或者当前区间与上一区间不重合，直接添加
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # 否则的话，我们就可以与上一区间进行合并
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged
```

### rust

```rust
impl Solution {
    pub fn merge(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut intervals = intervals;
        intervals.sort();

        let mut merged = vec![intervals[0].clone()];
        for interval in intervals {
            let n = merged.len() - 1;
            if merged[n][1] < interval[0] {
                merged.push(interval);
            } else {
                merged[n][1] = std::cmp::max(merged[n][1], interval[1])
            }
        }
        return merged;
    }
}
```

* 时间复杂度：$O(n\log{n})$，其中 $n$ 为区间的数量。除去排序的开销，我们只需要一次线性扫描，所以主要的时间开销是排序的 $O(n\log{n})$。
* 空间复杂度：$O(\log{n})$，其中 $n$ 为区间的数量。这里计算的是存储答案之外，使用的额外空间。$O(\log{n})$ 即为排序所需要的空间复杂度。
