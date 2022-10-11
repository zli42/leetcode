### [Merge Sorted Array](https://leetcode.cn/problems/merge-sorted-array/)

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

### [First Bad Version](https://leetcode.cn/problems/first-bad-version/)

```python
# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        left = 1
        right = n
        while left < right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left
```

* 时间复杂度：$O(\log n)$，其中 $n$ 是给定版本的数量。
* 空间复杂度：$O(1)$。我们只需要常数的空间保存若干变量。
