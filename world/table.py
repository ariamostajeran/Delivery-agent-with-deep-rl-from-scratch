
class Table:
    def __init__(self, grid, upper_right_col, upper_right_row):
        self.grid = grid
        self.cells =self._create_full_table(upper_right_col, upper_right_row)
        self.customers = 0
        
        self.shape = (max(cell[0] for cell in self.cells), max(cell[1] for cell in self.cells))

    def _create_full_table(self, upper_right_col, upper_right_row):
        '''
        Gets the full table assuming it is of rectangular shape. Due to how we explore the grid later,
        it will always find the upper right corner of the table.
        '''
        table = []
    
        i = upper_right_col
        j = upper_right_row + 1  #+1 to not add the corner two times


        #First explore the width
        while self.grid[i, upper_right_row] == 6:
            table.append((i, upper_right_row))
            i += 1

        #Explore the height
        while self.grid[upper_right_col, j] == 6:
            table.append((upper_right_col, j))
            j += 1

        #get remaining cells
        for k in range(upper_right_col+1,i):
            for l in range(upper_right_row+1, j):
                table.append((k,l))

        return table
    
    def __str__(self):
        '''
        Returns a string representation of the table.
        '''
        return f'Table[Customers: {self.customers},\n      Cells: {self.cells}]'
    
    def set_customers(self, customers):
        '''
        Sets the number of customers in the table
        '''
        if customers > self.shape[0]*self.shape[1]:
            raise ValueError(f'The number of costumers does not fit in a table of shape {self.shape}')
        else:
            self.customers = customers

