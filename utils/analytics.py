class Analytics:


    def __init__(self, database):

        self.database = database



    def get_statistics(self):


        history = self.database.get_history()


        total_cases = len(
            history
        )


        total_area = 0

        disasters = {}


        for row in history:


            disaster = row[2]

            area = row[4]


            try:

                total_area += float(area)

            except:

                pass



            if disaster in disasters:

                disasters[disaster] += 1

            else:

                disasters[disaster] = 1



        return {

            "cases": total_cases,

            "area": round(
                total_area,
                2
            ),

            "disasters": disasters

        }