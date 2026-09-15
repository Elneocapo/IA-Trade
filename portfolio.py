"""Simulador de cartera: no ejecuta operaciones reales."""

from dataclasses import dataclass


@dataclass
class Portfolio:
    cash: float
    shares: float = 0.0

    def buy_with_all_cash(self, price: float, commission: float = 0.001) -> None:
        if price <= 0 or self.cash <= 0:
            return
        budget = self.cash / (1 + commission)
        self.shares = budget / price
        self.cash = self.cash - budget * (1 + commission)

    def sell_all(self, price: float, commission: float = 0.001) -> None:
        if price <= 0 or self.shares <= 0:
            return
        gross = self.shares * price
        self.cash += gross * (1 - commission)
        self.shares = 0.0

    def value(self, price: float) -> float:
        return self.cash + self.shares * price
