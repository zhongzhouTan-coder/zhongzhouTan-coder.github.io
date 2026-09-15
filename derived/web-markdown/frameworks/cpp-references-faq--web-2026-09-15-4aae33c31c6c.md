---
kind: web-extraction
source_url: "https://isocpp.org/wiki/faq/references"
final_url: "https://isocpp.org/wiki/faq/references"
canonical_url: "https://isocpp.org/wiki/faq/references"
title: "Standard C++"
author: ""
published_at: ""
captured_at: "2026-09-15T07:41:08.796Z"
content_sha256: 4aae33c31c6c9712708957020beb4c1d9912c0aa72626380ace54a13e9a68155
renderer: http
extractor: "defuddle@0.13.0 + turndown@7.2.4"
---

[Show sidebar »](#sidebar)

-   [Sign In / Suggest an Article](https://isocpp.org/member/login)
-   [Register](https://isocpp.org/member/register)

[![Standard C++](https://isocpp.org/assets/images/cpp_logo.png)](https://isocpp.org/)
=====================================================================================

-   [Get Started!](https://isocpp.org/get-started)
-   [Tour](https://isocpp.org/tour)
-   [Core Guidelines](https://isocpp.org/guidelines)
-   [Super-FAQ](https://isocpp.org/faq)
-   [Standardization](https://isocpp.org/std)
-   [About](https://isocpp.org/about)

Features

[Current ISO C++ status](https://isocpp.org/std/status)

[ISO C++ standing documents](https://isocpp.org/std/standing-documents)

[Upcoming ISO C++ meetings](https://isocpp.org/std/meetings-and-participation/upcoming-meetings)

[Upcoming C++ conferences](https://isocpp.org/wiki/faq/conferences-worldwide)

[Compiler conformance status](http://en.cppreference.com/w/cpp/compiler_support)

Navigation

-   [FAQ Home](https://isocpp.org/wiki/faq)
-   [FAQ RSS Feed](https://isocpp.org/themes/wiki_themes/isocpp/rss/faq-revisions-rss.php)
-   [FAQ Help](https://isocpp.org/wiki/faq/editinfo)

Search this Wiki

Go to Page

Upcoming Events

[**Pure Virtual C++**](https://devblogs.microsoft.com/cppblog/save-the-date-pure-virtual-cpp-2026/)

July 21, Online

[**CppCon**](https://cppcon.org/)

September 12-18, Aurora, CO, USA

[**C++ Day 2026**](https://italiancpp.github.io/cppday26/)

October 17, Pavia, Italy

[**ISO C++ meeting - Búzios**](https://isocpp.org/files/papers/N5021.html)

November 16-21, Búzios, Rio De Janeiro, Brazil

[**Meeting C++**](http://meetingcpp.com/2026)

November 26-28, Berlin, Germany

[Wiki Home](https://isocpp.org/wiki/faq)   >   references

[View](https://isocpp.org/wiki/faq/references)

references
----------

References
----------

### What is a reference?

An alias (an alternate name) for an object.

References are frequently used for pass-by-reference:

```
void swap(int& i, int& j)
{
  int tmp = i;
  i = j;
  j = tmp;
}

int main()
{
  int x, y;
  // ...
  swap(x,y);
  // ...
}
```

Here `i` and `j` are aliases for main’s `x` and `y` respectively. In other words, `i` **is** `x` — not a pointer to `x`, nor a copy of `x`, but `x` itself. Anything you do to `i` gets done to `x`, and vice versa. This includes taking the address of it. The values of `&i` and `&x` are identical.

That’s how you should think of references as a programmer. Now, at the risk of confusing you by giving you a different perspective, here’s how references are implemented. Underneath it all, a reference `i` to object `x` is typically the machine address of the object `x`. But when the programmer says `i++`, the compiler generates code that increments `x`. In particular, the address bits that the compiler uses to find `x` are not changed. A C programmer will think of this as if you used the C style pass-by-pointer, with the syntactic variant of (1) moving the `&` from the caller into the callee, and (2) eliminating the `*`s. In other words, a C programmer will think of `i` as a macro for `(*p)`, where `p` is a pointer to `x` (e.g., the compiler automatically dereferences the underlying pointer; `i++` is changed to `(*p)++`; `i = 7` is automatically changed to `*p = 7`).

*Important note:* Even though a reference is often implemented using an address in the underlying assembly language, please do *not* think of a reference as a funny looking pointer to an object. A reference **is** the object, just with another name. It is neither a pointer to the object, nor a copy of the object. It **is** the object. There is no C++ syntax that lets you operate on the reference itself separate from the object to which it refers.

### What happens if you assign to a reference?

You change the state of the referent (the referent is the object to which the reference refers).

Remember: the reference **is** the referent, so changing the reference changes the state of the referent. In compiler writer lingo, a reference is an “lvalue” (something that can appear on the left hand side of an assignment `operator`).

### What happens if you return a reference?

The function call can appear on the left hand side of an assignment `operator`.

This ability may seem strange at first. For example, no one thinks the expression `f() = 7` makes sense. Yet, if `a` is an object of `class` `Array`, most people think that `a[i] = 7` makes sense even though `a[i]` is really just a function call in disguise (it calls `Array::operator[](int)`, which is the subscript `operator` for `class` `Array`).

```
class Array {
public:
  int size() const;
  float& operator[] (int index);
  // ...
};

int main()
{
  Array a;
  for (int i = 0; i < a.size(); ++i)
    a[i] = 7;    // This line invokes Array::operator[](int)
  // ...
}
```

### What does `object.method1().method2()` mean?

It *chains* these method calls, which is why this is called *method chaining*.

The first thing that gets executed is `object.method1()`. This returns some object, which might be a reference to `object` (i.e., `method1()` might end with `return *this;`), or it might be some other object. Let’s call the returned object `objectB`. Then `objectB` becomes the `this` object of `method2()`.

The most common use of method chaining is in the `iostream` library. E.g., `cout << x << y` works because `cout << x` is a function that returns `cout`.

A less common, but still rather slick, use for method chaining is in [the Named Parameter Idiom](https://isocpp.org/wiki/faq/ctors#named-parameter-idiom "FAQ: What is the 'Named Parameter Idiom'?").

### How can you reseat a reference to make it refer to a different object?

No way.

You can’t separate the reference from the referent.

Unlike a pointer, once a reference is bound to an object, it can **not** be “reseated” to another object. The reference isn’t a separate object. It has no identity. Taking the address of a reference gives you the address of the referent. Remember: the reference **is** its referent.

In that sense, a reference is similar to a [`const` pointer](https://isocpp.org/wiki/faq/const-correctness#const-ptr-vs-ptr-const "FAQ: What's the difference between 'const X* p', 'X* const p' and 'const X* const p'?") such as `int* const p` (as opposed to a [pointer to `const`](https://isocpp.org/wiki/faq/const-correctness#ptr-to-const "FAQ: What does 'const X* p' mean?") such as `const int* p`). But please don’t confuse references with pointers; they’re *very* different from the programmer’s standpoint.

### Why does C++ have both pointers and references?

C++ inherited pointers from C, so they couldn’t be removed without causing serious compatibility problems. References are useful for several things, but the direct reason they were introduced in C++ was to support operator overloading. For example:

```
    void f1(const complex* x, const complex* y) // without references
    {
        complex z = *x+*y;  // ugly
        // ...
    }

    void f2(const complex& x, const complex& y) // with references
    {
        complex z = x+y;    // better
        // ...
    }   
```

More generally, if you want to have both the functionality of pointers and the functionality of references, you need either two different types (as in C++) or two different sets of operations on a single type. For example, with a single type you need both an operation to assign to the object referred to and an operation to assign to the reference/pointer. This can be done using separate operators (as in Simula). For example:

```
    Ref<My_type> r :- new My_type;
    r := 7;         // assign to object
    r :- new My_type;   // assign to reference
```

Alternatively, you could rely on type checking (overloading). For example:

```
    Ref<My_type> r = new My_type;
    r = 7;          // assign to object
    r = new My_type;    // assign to reference
```

### When should I use references, and when should I use pointers?

Use references when you can, and pointers when you have to.

References are usually preferred over pointers whenever you don’t need [“reseating”](https://isocpp.org/wiki/faq/references#reseating-refs "FAQ: How can you reseat a reference to make it refer to a different object?"). This usually means that references are most useful in a class’s `public` interface. References typically appear on the skin of an object, and pointers on the inside.

The exception to the above is where a function’s parameter or return value needs a “sentinel” reference — a reference that does not refer to an object. This is usually best done by returning/taking a pointer, and giving the `nullptr` value this special significance ([references must always alias objects, not a dereferenced null pointer](https://isocpp.org/wiki/faq/references#refs-not-null "FAQ: What does it mean that a reference must refer to an object, not a dereferenced null pointer?")).

Note: Old line C programmers sometimes don’t like references since they provide reference semantics that isn’t explicit in the caller’s code. After some C++ experience, however, one quickly realizes this is a form of information hiding, which is an asset rather than a liability. E.g., programmers should write code in the language of the problem rather than the language of the machine.

### What does it mean that a reference must refer to an object, not a dereferenced null pointer?

It means this is illegal:

```
T* p = nullptr;
T& r = *p;  // illegal
```

*NOTE:* Please do **not** email us saying the above works on your particular version of your particular compiler. It’s still illegal. The C++ language, as defined by the C++ standard, says it’s illegal; that makes it illegal. The C++ standard does not require a diagnostic for this particular error, which means your particular compiler is not obliged to notice that `p` is `nullptr` or to give an error message, but it’s still illegal. The C++ language also does not require the compiler to generate code that would blow up at runtime. In fact, your particular version of your particular compiler may, or may not, generate code that you think makes sense if you do the above. But that’s the point: since the compiler is not required to generate sensible code, you don’t know what the compiler will do. So *please do not* email us saying your particular compiler generates good code; we don’t care. It’s still illegal. See the C++ standard for more, for example, C++ 2014 section 8.3.2 \[dcl.ref\] p5.

By way of example and **not** by way of limitation, some compilers do optimize `nullptr` tests since they “know” all references refer to real objects — that references are never (legally) a dereferenced `nullptr`. That can cause a compiler to optimize away the following test:

```
// ...the above code...
T* p2 = &r;
if (p2 == nullptr) {
  // ...
}
```

As stated above, this is just an *example* of the *sort* of thing your compiler might do based on the language rule that says a reference must refer to a valid object. Do *not* limit your thinking to the above example; the message of this FAQ is that the compiler is not required to do something sensible if you violate the rules. So don’t violate the rules.

> *Patient: “Doctor, doctor, my eye hurts when I poke it with a spoon.”*
> 
> *Doctor: “Don’t poke it, then.”*

### What is a *handle* to an object? Is it a pointer? Is it a reference? Is it a pointer-to-a-pointer? What is it?

The term *handle* is used to mean any technique that lets you get to another object — a generalized pseudo-pointer. The term is (intentionally) ambiguous and vague.

Ambiguity is actually an asset in certain cases. For example, during early design you might not be ready to commit to a specific representation for the handles. You might not be sure whether you’ll want simple pointers vs. references vs. pointers-to-pointers vs. references-to-pointers vs. integer indices into an array vs. strings (or other key) that can be looked up in a hash-table (or other data structure) vs. database keys vs. some other technique. If you merely know that you’ll need some sort of thingy that will uniquely identify and get to an object, you call the thingy a Handle.

So if your ultimate goal is to enable a glop of code to uniquely identify/look-up a specific object of some class `Fred`, you need to pass a `Fred` handle into that glop of code. The handle might be a string that can be used as a key in some well-known lookup table (e.g., a key in a [`std::map<std::string,Fred>`](https://isocpp.org/wiki/faq/containers#associative-array "FAQ: How can I make a perl-like associative array in C++?") or a [`std::map<std::string,Fred*>`](https://isocpp.org/wiki/faq/containers#associative-array "FAQ: How can I make a perl-like associative array in C++?")), or it might be an integer that would be an index into some well-known array (e.g., [`Fred* array = new Fred[maxNumFreds]`](https://isocpp.org/wiki/faq/freestore-mgmt#allocate-array "FAQ: How do I allocate / unallocate an array of things?")), or it might be a simple `Fred*`, or it might be something else.

Novices often think in terms of pointers, but in reality there are downside risks to using raw pointers. E.g., what if the `Fred` object needs to move? How do we know when it’s safe to `delete` the `Fred` objects? What if the `Fred` object needs to (temporarily) get serialized on disk? etc., etc. Most of the time we add more layers of indirection to manage situations like these. For example, the handles might be `Fred**`, where the pointed-to `Fred*` pointers are guaranteed to never move but when the `Fred` objects need to move, you just update the pointed-to `Fred*` pointers. Or you make the handle an integer then have the `Fred` objects (or pointers to the `Fred` objects) looked up in a table/array/whatever. Or whatever.

The point is that we use the word Handle when we don’t yet know the details of what we’re going to do.

Another time we use the word Handle is when we want to be vague about what we’ve already done (sometimes the term *magic cookie* is used for this as well, as in, “The software passes around a *magic cookie* that is used to uniquely identify and locate the appropriate `Fred` object”). The reason we (sometimes) want to be vague about what we’ve already done is to minimize the ripple effect if/when the specific details/representation of the handle change. E.g., if/when someone changes the handle from a string that is used in a lookup table to an integer that is looked up in an array, we don’t want to go and update a zillion lines of code.

To further ease maintenance if/when the details/representation of a handle changes (or to generally make the code easier to read/write), we often encapsulate the handle in a class. This class often [overloads operators](https://isocpp.org/wiki/faq/operator-overloading "Section: Operator Overloading") `operator->` and `operator*` (since the handle acts like a pointer, it might as well look like a pointer).

### Should I use call-by-value or call-by-reference?

(Note: This FAQ needs to be updated for C++11.)

That depends on what you are trying to achieve:

-   If you want to change the object passed, call by reference or use a pointer; e.g., `void f(X&);` or `void f(X*);`.
-   If you don’t want to change the object passed and it is big, call by const reference; e.g., `void f(const X&);`.
-   Otherwise, call by value; e.g. `void f(X);`.

*What does “big” mean?* Anything larger than a couple of words.

*Why would you want to change an argument?* Well, often we have to, but often we have an alternative: produce a new value. Consider:

```
    void incr1(int& x); // increment
    int incr2(int x);   // increment

    int v = 2;
    incr1(v);   // v becomes 3
    v = incr2(v);   // v becomes 4
```

For a reader, `incr2()` is likely easier to understand. That is, `incr1()` is more likely to lead to mistakes and errors. So, we should prefer the style that returns a new value over the one that modifies a value as long as the creation and copy of a new value isn’t expensive.

*What if you do want to change the argument, should you use a pointer or use a reference?* If passing “not an object” (e.g., a null pointer) is acceptable, using a pointer makes sense. One style is to use a pointer when you want to modify an object because in some contexts that makes it easier to spot that a modification is possible.

Note also that a call of a member function is essentially a call-by-reference on the object, so we often use member functions when we want to modify the value/state of an object.

### Why is `this` not a reference?

Because `this` was introduced into C++ (really into C with Classes) before references were added. Also, Stroustrup chose `this` to follow Simula usage, rather than the (later) Smalltalk use of `self`.

Please [Login](https://isocpp.org/member/login) to submit a recommendation.

If you don’t have an account, you can [register for free.](https://isocpp.org/member/register)

© [Copyright](https://isocpp.org/home/terms-of-use) 2026 Standard C++ Foundation. All rights reserved.

-   [Terms of Use](https://isocpp.org/home/terms-of-use)
-   [Privacy Policy](https://isocpp.org/home/privacy-policy)

Hosted by [DigitalOcean](https://m.do.co/c/32f291566cf7)

M20o93H7pQ09L8X1t49cHY01Z5j4TT91fGfr
