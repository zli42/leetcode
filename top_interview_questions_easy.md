# Top Interview Questions Easy Collection

## Array

### [Remove Duplicates from Sorted Array](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        j = 0
        for i in range(len(nums)):
            if nums[i] != nums[j]:
                j += 1
                nums[j] = nums[i]
        return j + 1
```

* 时间复杂度：$O(n)$，其中 $n$ 是数组的长度。快指针和慢指针最多各移动 $n$ 次。
* 空间复杂度：$O(1)$。只需要使用常数的额外空间。

### [Best Time to Buy and Sell Stock II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i - 1]
            if diff > 0:
                profit += diff
        return profit
```

* 时间复杂度：$O(n)$，其中 $n$ 为数组的长度。我们只需要遍历一次数组即可。
* 空间复杂度：$O(1)$。只需要常数空间存放若干变量。

### [Rotate Array](https://leetcode.cn/problems/rotate-array/)

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k %= n
        self.reverse(nums, 0, n - 1)
        self.reverse(nums, 0, k - 1)
        self.reverse(nums, k, n - 1)

    def reverse(self, nums, start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
```

* 时间复杂度：$O(n)$，其中 $n$ 为数组的长度。每个元素被翻转两次，一共 $n$ 个元素，因此总时间复杂度为 $O(2n)=O(n)$。
* 空间复杂度：$O(1)$。

### [Contains Duplicate](https://leetcode.cn/problems/contains-duplicate/)

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        s = set()
        for num in nums:
            if num in s:
                return True
            s.add(num)
        return False
```

* 时间复杂度：$O(N)$，其中 $N$ 为数组的长度。
* 空间复杂度：$O(N)$，其中 $N$ 为数组的长度。

### [Single Number](https://leetcode.cn/problems/single-number/)

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res ^= num
        return res
```

* 时间复杂度：$O(n)$，其中 $n$ 是数组长度。只需要对数组遍历一次。
* 空间复杂度：$O(1)$。