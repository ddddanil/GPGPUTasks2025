


  r0 1 2 3 4 5
 l -\
 0 x|x x x x x
    \---\
 1 x x x|x x x
        |
 2 x x x|x x x
        \---\
 3 x x x x x|x
            \-\
 4 x x x x x x|

diag 0 : (0,0,d), (0,0,r)
diag 1 : (0,1,d), (0,1,r), (1,0,d), (1,0,r)

  r0 1 2 3 4 5
 l 1 3 5 7 9 X
 00x2x4x6x8x9x9
   1 3 5 7 8 8
 10x2x4x6x7x7x7
   1 3 5 6 6 6
 20x2x4x5x5x5x5
   1 3 4 4 4 4
 30x2x3x3x3x3x3
   1 2 2 2 2 2 
 40x1x1x1x1x1x1 
   0 0 0 0 0 0

widths: 2 3 6 8 10 11 10 8 6 4 2
path: 1 2 3 5 6 4 3 3 2 1 e


down
 r 0 1 2 3 4 5

l  1 3 5 7 9  
0 . . . . .    
 0 1 3 5 7    
1 . . . .      
 1 1 3 5      
2 . . .        
 2 1 3        
3 . .          
 3 1           
4 .             
 4            

 r 0 1 2 3 4 5

l            X
0           . .
           8 8
1         . . .
         6 6 6
2       . . . .
       4 4 4 4
3     . . . . .
     2 2 2 2 2 
4   . . . . . . 
   0 0 0 0 0 0
  5 6 7 8 9 X

left
 r 0 1 2 3 4 5

l  . . . . .  
0 0 2 4 6 8    
 0 . . . .    
1 0 2 4 6      
 1 . . .      
2 0 2 4        
 2 . .        
3 0 2          
 3 .           
4 0             
 4            

 r 0 1 2 3 4 5

l            .
0           9 9
           . .
1         7 7 7
         . . .
2       5 5 5 5
       . . . .
3     3 3 3 3 3
     . . . . . 
4   1 1 1 1 1 1 
   . . . . . .
  5 6 7 8 9 X

diag 2 (0,3) (1,2) (2,1) (3,0)


  r0 1
 l 1 3
 00x2x3
   1 2
 10x1x1
   0 0

	 5 7
    r0 1
   l 1 3
 5 00T2T3
     1 2
16 10F1F1
     0 0

	 5 7
    r0 1
   l    
 5 0|T T 
    \---\
16 1 F F|
        
path 0 1 2 1

	         1
	   5 5 7 6
	  r0 1 2 3  
	 l 1 3 5 7  
   4 00x2x4x6x7
	   1 3 5 6  
   5 10x2x4x5x5
	   1 3 4 4  
   8 20x2x3x3x3
	   1 2 2 2  
  13 30x1x1x1x1
	   0 0 0 0   

	
			 1 
	   5 5 7 6 
	  r0 1 2 3  
	 l           
   4 0|T T T T  
	  |          
   5 1|T T T T  
	  \-----\    
   8 2 F F F|T  
	        |    
  13 3 F F F|T  
            \-
	              
path 0 0 1 3 4 3 1 0

			 1 
	   5 5 7 6 
	  r0 1 2 3  
	 l t         
   4 0fT T T T  
	   t         
   5 1fTtTtTtT  
	   f f f t   
   8 2 F F FfT  
	         t   
  13 3 F F FfTt 
             f

1: f t
    T

      T
1: f t t t
    T   T

      T   T
2: f f t t t t
    F   T   T

      F   T   T
3: f f f f t t t t
    F   F   T   T

    F   F   T   T
4: f f f f f t t t
      F   F   T
