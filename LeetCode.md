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

# [单词搜索](https://leetcode.cn/problems/word-search/solution/dan-ci-sou-suo-by-leetcode-solution/)

## 回溯

### python

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def check(i: int, j: int, k: int) -> bool:
            if board[i][j] != word[k]:
                return False
            if k == len(word) - 1:
                return True
            
            visited.add((i, j))
            result = False
            for di, dj in directions:
                newi, newj = i + di, j + dj
                if 0 <= newi < len(board) and 0 <= newj < len(board[0]):
                    if (newi, newj) not in visited:
                        if check(newi, newj, k + 1):
                            result = True
                            break
            
            visited.remove((i, j))
            return result

        h, w = len(board), len(board[0])
        visited = set()
        for i in range(h):
            for j in range(w):
                if check(i, j, 0):
                    return True
        
        return False
```

## rust

```rust
use std::collections::HashSet;

impl Solution {
    pub fn exist(board: Vec<Vec<char>>, word: String) -> bool {
        fn check(
            i: usize,
            j: usize,
            k: usize,
            board: &Vec<Vec<char>>,
            word: &Vec<char>,
            visited: &mut HashSet<(usize, usize)>,
        ) -> bool {
            if board[i][j] != word[k] {
                return false;
            }
            if k == word.len() - 1 {
                return true;
            }

            visited.insert((i, j));
            let mut result = false;
            let directions: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];
            for (di, dj) in directions {
                let newi = i + di as usize;
                let newj = j + dj as usize;
                let newk = k + 1;
                if 0 <= newi && newi < board.len() && 0 <= newj && newj < board[0].len() {
                    if !visited.contains(&(newi, newj)) {
                        if check(newi, newj, newk, board, word, visited) {
                            result = true;
                            break;
                        }
                    }
                }
            }
            visited.remove(&(i, j));
            return result;
        }

        let word = word.chars().collect::<Vec<char>>();
        let mut visited = HashSet::new();
        for i in 0..board.len() {
            for j in 0..board[0].len() {
                if check(i, j, 0, &board, &word, &mut visited) {
                    return true;
                }
            }
        }
        return false;
    }
}
```

* 时间复杂度：一个非常宽松的上界为 $O(MN \cdot 3^L)$，其中 $M$, $N$ 为网格的长度与宽度，$L$ 为字符串 `word` 的长度。在每次调用函数 `check` 时，除了第一次可以进入 $4$ 个分支以外，其余时间我们最多会进入 $3$ 个分支（因为每个位置只能使用一次，所以走过来的分支没法走回去）。由于单词长为 $L$，故 `check(i,j,0)` 的时间复杂度为 $O(3^L)$，而我们要执行 $O(MN)$ 次检查。然而，由于剪枝的存在，我们在遇到不匹配或已访问的字符时会提前退出，终止递归流程。因此，实际的时间复杂度会远远小于 $O(MN \cdot 3^L)$。
* 空间复杂度：$O(MN)$。我们额外开辟了 $O(MN)$ 的 `visited` 数组，同时栈的深度最大为 $O(\min(L, MN))$。

# [解码方法](https://leetcode.cn/problems/decode-ways/)

## 动态规划

### python

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        pp, pre, cur = 0, 1, 0
        for i in range(1, n + 1):
            cur = 0
            if s[i - 1] != '0':
                cur += pre
            if i > 1 and s[i - 2] != '0' and int(s[i-2:i]) <= 26:
                cur += pp
            pp, pre = pre, cur
        return cur
```

### rust

```rust
impl Solution {
    pub fn num_decodings(s: String) -> i32 {
        let s = s.chars().collect::<Vec<char>>();
        let mut pp = 0;
        let mut pre = 1;
        let mut cur = 0;
        for i in 1..s.len() + 1 {
            cur = 0;
            if s[i - 1] != '0' {
                cur += pre;
            }
            if i > 1
                && s[i - 2] != '0'
                && s[i - 2..i]
                    .iter()
                    .collect::<String>()
                    .parse::<i32>()
                    .unwrap()
                    <= 26
            {
                cur += pp;
            }
            pp = pre;
            pre = cur;
        }
        cur
    }
}
```

* 时间复杂度：$O(n)$，其中 $n$ 是字符串 `s` 的长度。
* 空间复杂度：$O(1)$。
