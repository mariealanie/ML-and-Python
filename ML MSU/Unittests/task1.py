def hello(x=None):
   if x == None or x == "":
      return "Hello!"
   return f"Hello, {x}!"

def int_to_roman(num):
   val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
   syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
   rm_num = ''
   i = 0
   while num > 0:
      for _ in range(num // val[i]):
         rm_num += syms[i]
      num %= val[i]
      i += 1
   return rm_num

def longest_common_prefix(stri):
   cor_stri = [s.strip() for s in stri if s.strip()]
   if len(cor_stri) == 0:
      return ""
   pr = cor_stri[0]
   for s in cor_stri[1:]:
     while not s.startswith(pr):
         pr = pr[:-1]
         if not pr:
            return ""
   return pr

class BankCard:
   def __init__(self, total_sum, balance_limit=-1):
      self.total_sum = total_sum
      self.balance_limit = balance_limit

   def __call__(self, sum_spent):
      if self.total_sum - sum_spent < 0:
         print(f"Not enough money to spend {sum_spent} dollars.")
         raise ValueError
      self.total_sum -= sum_spent
      print(f"You spent {sum_spent} dollars.")

   def __add__(self, another_card):
      if not isinstance(another_card, BankCard):
         print("You can add only BankCard instance.")
         raise ValueError
      new_total_sum = self.total_sum + another_card.total_sum
      if self.balance_limit == -1 or another_card.balance_limit == -1:
         new_balance_limit = -1
      else:
         new_balance_limit = max(self.balance_limit, another_card.balance_limit)
      return BankCard(new_total_sum, new_balance_limit)

   @property
   def balance(self):
      if self.balance_limit == 0:
         print("Balance check limits exceeded.")
         raise ValueError
      elif self.balance_limit != -1:
         self.balance_limit -= 1
      return self.total_sum

   def __str__(self):
      return "To learn the balance call balance."

   def put(self, sum_put):
      self.total_sum += sum_put
      print(f"You put {sum_put} dollars.")

def primes():
   yield 2
   primes_list = [2]
   num = 3
   while True:
      is_prime = True
      for prime in primes_list:
         if prime * prime > num:
            break
         if num % prime == 0:
            is_prime = False
            break
      if is_prime:
         primes_list.append(num)
         yield num
      num += 2
longest_common_prefix(['   ', ' ', ''])                                   
