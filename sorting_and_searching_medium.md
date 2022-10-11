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
