/*
  Merge sort
  Time: O(Nlg(N))	1/2NlgN ~ NlgN
  Space: O(N)	6NlgN
*/

import java.util.Arrays;

public class MergeSort{
	private static int[] temp;
	
	public static void sort(int[] a){
		temp = new int[a.length];
		sort(a, 0, a.length - 1);
	}
	
	private static void sort(int[] a, int lo, int hi){
		if(hi <= lo){
			return;
		}
		int mid = lo + (hi - lo) / 2;
		sort(a, lo, mid);
		sort(a, mid + 1, hi);
		merge(a, lo, mid, hi);
	}
	
	private static void merge(int[] a, int lo, int mid, int hi){
		for(int k = lo; k <= hi; k++){
			temp[k] = a[k];
		}
		int i = lo, j = mid + 1;
		for(int k = lo; k <= hi; k++){
			if(i > mid){
				a[k] = temp[j++];
			}else if(j > hi){
				a[k] = temp[i++];
			}else if(temp[j] < temp[i]){
				a[k] = temp[j++];
			}else{
				a[k] = temp[i++];
			}
		}
	}
	
	public static void main(String[] args){
		int[] a = {2, 5, 8, 6, 4, 9, 0, 6, 7, 1, 3};
        MergeSort.sort(a);
		System.out.println(Arrays.toString(a));
	}
}