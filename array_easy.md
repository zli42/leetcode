### [Remove Duplicates from Sorted Array](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)

python
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

rust
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

### [Best Time to Buy and Sell Stock II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

python
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

python
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

python
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

位异或运算
1. 任何数和 $0$ 做异或运算，结果仍然是原来的数，即 $a \oplus 0=a$。
2. 任何数和其自身做异或运算，结果是 $0$，即 $a \oplus a=0$。
3. 异或运算满足交换律和结合律，即 $a \oplus b \oplus a=b \oplus a \oplus a=b \oplus (a \oplus a)=b \oplus0=b$。

python
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

### [Intersection of Two Arrays II](https://leetcode.cn/problems/intersection-of-two-arrays-ii/)

python
```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if len(nums1) > len(nums2):
            return self.intersect(nums2, nums1)

        m = dict()
        for num in nums1:
            m[num] = m.setdefault(num, 0) + 1

        intersection = list()
        for num in nums2:
            if m.get(num, 0) > 0:
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

* 时间复杂度：$O(m+n)$，其中 $m$ 和 $n$ 分别是两个数组的长度。需要遍历两个数组并对哈希表进行操作，哈希表操作的时间复杂度是 $O(1)$，因此总时间复杂度与两个数组的长度和呈线性关系。
* 空间复杂度：$O(\min(m,n))$，其中 $m$ 和 $n$ 分别是两个数组的长度。对较短的数组进行哈希表的操作，哈希表的大小不会超过较短的数组的长度。为返回值创建一个数组 `intersection`，其长度为较短的数组的长度。


### [Plus One](https://leetcode.cn/problems/plus-one/)

python
```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        n = len(digits)
        for i in range(n - 1, -1, -1):
            if digits[i] != 9:
                digits[i] += 1
                for j in range(i + 1, n):
                    digits[j] = 0
                return digits

        # digits 中所有的元素均为 9
        return [1] + [0] * n
```

* 时间复杂度：$O(n)$，其中 $n$ 是数组 `digits` 的长度。
* 空间复杂度：$O(1)$。返回值不计入空间复杂度。

### [Move Zeroes](https://leetcode.cn/problems/move-zeroes/)

python
```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1
```

* 时间复杂度：$O(n)$，其中 $n$ 为序列长度。每个位置至多被遍历两次。
* 空间复杂度：$O(1)$。只需要常数的空间存放若干变量。

### [Two Sum](https://leetcode.cn/problems/two-sum/)

python
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        res = dict()
        for i, num in enumerate(nums):
            if num in res:
                return [res[num], i]
            res[target - num] = i
        return list()
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

* 时间复杂度：$O(N)$，其中 $N$ 是数组中的元素数量。对于每一个元素 x，我们可以 $O(1)$ 地寻找 target - x。
* 空间复杂度：$O(N)$，其中 $N$ 是数组中的元素数量。主要为哈希表的开销。

### [Valid Sudoku](https://leetcode.cn/problems/valid-sudoku/)

python
```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row = [set() for _ in range(9)]
        col = [set() for _ in range(9)]
        cell = [set() for _ in range(9)]
        for i in range(9):
            for j in range(9):
                n = board[i][j]
                if n == ".":
                    continue
                if n in row[i] or n in col[j] or n in cell[i // 3 * 3 + j // 3]:
                    return False
                row[i].add(n)
                col[j].add(n)
                cell[i // 3 * 3 + j // 3].add(n)
        return True
```

* 时间复杂度：$O(1)$。数独共有 $81$ 个单元格，只需要对每个单元格遍历一次即可。
* 空间复杂度：$O(1)$。由于数独的大小固定，因此哈希表的空间也是固定的。

### [Rotate Image](https://leetcode.cn/problems/rotate-image/)

python
```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
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

* 时间复杂度：$O(N^2)$，其中 $N$ 是 `matrix` 的边长。对于每一次翻转操作，我们都需要枚举矩阵中一半的元素。
* 空间复杂度：$O(1)$。为原地翻转得到的原地旋转。
