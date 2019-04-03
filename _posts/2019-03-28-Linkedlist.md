---
layout:     post
title:      (Leetcode Topic 1) Linkedlist Summarize
subtitle:    
date:       2019-03-28
author:     Lu Zhang
header-img: img/images/img-linklist.jpg
catalog: true
tags:
    - Leetcode
    - Java
---
## Description
The linklist is the foundamental data structure 
Basic stucture :
``` java
//Definition for singly-linked list.
 public class ListNode {
     int val;
     ListNode next;
     ListNode(int x) { val = x; }
  }
```
Type of Linkedlist: 
1. Single-linked list :  head, tail.next  = null
2. Circle: tail.next = head
3. Double:  we have head, tail pointer for list, also for every node
4. Customize Structure
``` java
// Double
public class ListNode{
    int val;
    ListNode pre;
    ListNode next;
    ListNode(int x){
        val = x ;
    }
}
```
Methods for the linkedlist part problems: 
1. Write basic model like, insert, delete, reverse and also need to be really expert 
2. Combine the basic model to solve the whole big problem

### Prob1. 876. Middle of the Linked List
[https://leetcode.com/problems/middle-of-the-linked-list/]

1. hashset 
2. o(n)  visited 
3. slow  and fast  slow = slow.next  fast = fast.next.next 
The use the M3: we can find 1/3, 1/4, 1/n 


### Prob2. 141. Linked List Cycle 
[https://leetcode.com/problems/linked-list-cycle/]

 slow.next  fast.next.next 
         slow == fast 
    If there is a circle, then the two pointers will at end meet at some one point, because both of them are trapped in the circle

### Prob3. 142. Linked List Cycle II  
[https://leetcode.com/problems/linked-list-cycle-ii/]

Find the cycle start point 
Use a extra list to mark the visited and and then return the first visited node
o(n) visited 


### Prob4. Insert a node into sorted LinkedList 
Steps:
  a. Find the node 
  b. Insert node 

### Prob5.  203. Remove Linked List Elements 
[https://leetcode.com/problems/remove-linked-list-elements/]

For delete, the important part is that we need to use a tmp node to mark the target.next first, incase we loose the information 
** Solution **
    aa. with head  prev = head, prev.next.val == target.val, prev.next = prev.next.next
    bb. without head  change the value, (problems:  reference is wrong)
```java 
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        if(head==null ) return null;
        ListNode dummynode = new ListNode(0);
        dummynode.next = head;
        ListNode prev =dummynode;
        while(prev.next!=null){
            while( prev.next!=null && prev.next.val!=val){
            prev = prev.next;
            }
        if(prev.next != null){
             prev.next= prev.next.next;
            }
        }
        return dummynode.next;
    }
}
```

### Prob6. 237. Delete Node in a Linked List  
[https://leetcode.com/problems/delete-node-in-a-linked-list/]
```java
class Solution {
    public void deleteNode(ListNode node) {
        ListNode next = node.next;
        node.val = next.val;
        node.next = next.next;
    }
}
```

### Prob7. 83. Remove Duplicates from Sorted List 
[https://leetcode.com/problems/remove-duplicates-from-sorted-list/]

```java 

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
         // with dummynode 
        if(head == null) return head;
        ListNode hh = new ListNode(0);
        hh.next = head;
        ListNode pre = head;
        while(pre.next!=null){
            if(pre.val == pre.next.val){
                pre.next = pre.next.next;
            }
            else{
                pre = pre.next;
            }
        }
        return hh.next;
        
    }
}
```

### Prob8. 21. Merge Two Sorted Lists 
[https://leetcode.com/problems/merge-two-sorted-lists/]

```java

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode prehead = new ListNode(-1);
        ListNode prev = prehead;
        while(l1!=null&& l2!= null){
            if(l1.val<l2.val){
                prev.next = l1;
                l1= l1.next;
            }
            else{
                prev.next = l2;
                l2 = l2.next;
            }
            prev = prev.next;
        }
        prev.next = l1 ==null ?l2:l1;
        
        return prehead.next;
    }
}
```
### Prob9. 143. Reorder List 
[https://leetcode.com/problems/reorder-list/]

```java

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public void reorderList(ListNode head) {
        // find the middle
        ListNode mid =  findMid(head);
        if(mid!= null){
              // find the next
        ListNode h1 = head;
        ListNode h2 = mid.next;
        mid.next = null;
        //reverse the link
        ListNode rh2 = reverseLink(h2);
        //merge h1 and h2
        head = mergeList(h1,rh2); 
        }
    }
    
    private ListNode mergeList(ListNode h1, ListNode h2){        
        ListNode tmp1 = h1;
        ListNode tmp2 = h2;
        while(h1!=null && h2!=null){
            tmp1 = h1.next;
            tmp2 = h2.next;
            h1.next = h2;
            h2.next = tmp1;
            h1 = h1.next.next; 
            h2 = tmp2;         
        } 
        return h1;
    }
    
    private ListNode findMid(ListNode head){
        if(head == null) return null;
        ListNode slow = head;
        ListNode fast = head;
        while(fast.next!=null && fast.next.next!=null){
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
    
    private ListNode reverseLink(ListNode head){
        if(head == null) return null;
        ListNode pos = null;
        ListNode prev = head;
        ListNode tmp =null;
        while(prev!= null){
            System.out.println(prev.val);
            tmp = prev.next;
            prev.next = pos;
            pos = prev;
            prev = tmp;         
        }
        return pos;
    } 
}
```


